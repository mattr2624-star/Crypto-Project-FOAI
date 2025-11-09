"""
Feature engineering pipeline: Kafka consumer that computes windowed features
from raw tick data and publishes to ticks.features topic.
"""

import argparse
import json
import logging
from datetime import datetime
from collections import deque
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from kafka import KafkaConsumer, KafkaProducer
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureComputer:
    """Computes windowed features from streaming tick data."""
    
    def __init__(self, 
                 window_sizes: list = [30, 60, 300],  # seconds
                 max_buffer_size: int = 10000):
        """
        Initialize feature computer with sliding windows.
        
        Args:
            window_sizes: List of window sizes in seconds
            max_buffer_size: Maximum number of ticks to keep in memory
        """
        self.window_sizes = window_sizes
        self.max_buffer_size = max_buffer_size
        
        # Buffers for different data types
        self.ticks_buffer = deque(maxlen=max_buffer_size)
        self.prices_buffer = deque(maxlen=max_buffer_size)
        self.timestamps_buffer = deque(maxlen=max_buffer_size)
        
        logger.info(f"FeatureComputer initialized with windows: {window_sizes}s")
    
    def _get_midprice(self, tick: Dict[str, Any]) -> Optional[float]:
        """Extract midprice from tick data."""
        try:
            # Try direct price field first
            if 'price' in tick:
                return float(tick['price'])
            
            # Calculate from bid/ask
            best_bid = float(tick.get('best_bid', 0))
            best_ask = float(tick.get('best_ask', 0))
            
            if best_bid > 0 and best_ask > 0:
                return (best_bid + best_ask) / 2.0
            return None
        except (ValueError, TypeError):
            return None
    
    def _get_spread(self, tick: Dict[str, Any]) -> Optional[float]:
        """Calculate bid-ask spread."""
        try:
            best_bid = float(tick.get('best_bid', 0))
            best_ask = float(tick.get('best_ask', 0))
            
            if best_bid > 0 and best_ask > 0:
                return best_ask - best_bid
            return None
        except (ValueError, TypeError):
            return None
    
    def _get_spread_bps(self, tick: Dict[str, Any]) -> Optional[float]:
        """Calculate bid-ask spread in basis points."""
        try:
            best_bid = float(tick.get('best_bid', 0))
            best_ask = float(tick.get('best_ask', 0))
            midprice = (best_bid + best_ask) / 2.0
            
            if midprice > 0:
                spread = best_ask - best_bid
                return (spread / midprice) * 10000  # basis points
            return None
        except (ValueError, TypeError):
            return None
    
    def add_tick(self, tick: Dict[str, Any]):
        """Add a new tick to the buffer."""
        self.ticks_buffer.append(tick)
        
        # Extract and store price and timestamp
        midprice = self._get_midprice(tick)
        if midprice:
            self.prices_buffer.append(midprice)
        
        # Parse timestamp - handle both 'time' and 'timestamp' fields
        timestamp_str = tick.get('timestamp', tick.get('time'))
        if timestamp_str:
            try:
                ts = pd.to_datetime(timestamp_str)
                self.timestamps_buffer.append(ts)
            except:
                self.timestamps_buffer.append(pd.Timestamp.now())
        else:
            self.timestamps_buffer.append(pd.Timestamp.now())
    
    def _get_window_data(self, window_seconds: int) -> tuple:
        """
        Get data within the specified time window.
        
        Returns:
            (prices_in_window, ticks_in_window)
        """
        if len(self.timestamps_buffer) < 2:
            return [], []
        
        current_time = self.timestamps_buffer[-1]
        cutoff_time = current_time - pd.Timedelta(seconds=window_seconds)
        
        prices_in_window = []
        ticks_in_window = []
        
        for i, ts in enumerate(self.timestamps_buffer):
            if ts >= cutoff_time:
                if i < len(self.prices_buffer):
                    prices_in_window.append(self.prices_buffer[i])
                if i < len(self.ticks_buffer):
                    ticks_in_window.append(self.ticks_buffer[i])
        
        return prices_in_window, ticks_in_window
    
    def compute_features(self, current_tick: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute all features for the current tick.
        
        Returns:
            Dictionary of features
        """
        features = {
            'timestamp': current_tick.get('timestamp', current_tick.get('time')),
            'product_id': current_tick.get('product_id', ''),
            'price': self._get_midprice(current_tick),
            'best_bid': float(current_tick.get('best_bid', 0)) if current_tick.get('best_bid') else None,
            'best_ask': float(current_tick.get('best_ask', 0)) if current_tick.get('best_ask') else None,
            'spread': self._get_spread(current_tick),
            'spread_bps': self._get_spread_bps(current_tick),
        }
        
        # Compute windowed features
        for window in self.window_sizes:
            prices, ticks = self._get_window_data(window)
            
            if len(prices) > 1:
                # Returns
                returns = np.diff(prices) / prices[:-1]
                features[f'return_mean_{window}s'] = float(np.mean(returns))
                features[f'return_std_{window}s'] = float(np.std(returns))
                features[f'return_min_{window}s'] = float(np.min(returns))
                features[f'return_max_{window}s'] = float(np.max(returns))
                
                # Price statistics
                features[f'price_mean_{window}s'] = float(np.mean(prices))
                features[f'price_std_{window}s'] = float(np.std(prices))
                
                # Trade intensity (tick count)
                features[f'tick_count_{window}s'] = len(ticks)
            else:
                # Not enough data for this window
                features[f'return_mean_{window}s'] = None
                features[f'return_std_{window}s'] = None
                features[f'return_min_{window}s'] = None
                features[f'return_max_{window}s'] = None
                features[f'price_mean_{window}s'] = None
                features[f'price_std_{window}s'] = None
                features[f'tick_count_{window}s'] = 0
        
        return features


class FeaturePipeline:
    """Main feature pipeline: consume from Kafka, compute features, publish and save."""
    
    def __init__(self,
                 input_topic: str = 'ticks.raw',
                 output_topic: str = 'ticks.features',
                 bootstrap_servers: str = 'localhost:9092',
                 output_file: str = 'data/processed/features.parquet',
                 window_sizes: list = [30, 60, 300],
                 create_kafka: bool = True):
        """Initialize the feature pipeline."""
        
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.output_file = output_file
        
        # Create output directory
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize Kafka consumer/producer if requested (set False for unit tests)
        if create_kafka:
            self.consumer = KafkaConsumer(
                input_topic,
                bootstrap_servers=bootstrap_servers,
                value_deserializer=self._safe_json_deserializer,
                auto_offset_reset='earliest',
                group_id='feature-pipeline',
                enable_auto_commit=True
            )

            # Initialize Kafka producer
            self.producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
        else:
            # For tests we leave consumer/producer as None (or user may inject fakes)
            self.consumer = None
            self.producer = None
        
        # Initialize feature computer
        self.feature_computer = FeatureComputer(window_sizes=window_sizes)
        
        # Buffer for batch writing to parquet
        self.features_batch = []
        self.batch_size = 100
        
        logger.info(f"FeaturePipeline initialized")
        logger.info(f"  Input topic: {input_topic}")
        logger.info(f"  Output topic: {output_topic}")
        logger.info(f"  Output file: {output_file}")
    
    @staticmethod
    def _safe_json_deserializer(message_bytes):
        """Safely deserialize JSON, handling errors gracefully."""
        if message_bytes is None or len(message_bytes) == 0:
            logger.warning("Received empty message, skipping")
            return None
        try:
            decoded = message_bytes.decode('utf-8').strip()
            if not decoded:
                logger.warning("Received whitespace-only message, skipping")
                return None
            return json.loads(decoded)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error: {e}. Raw bytes (first 100): {message_bytes[:100]}")
            return None
        except UnicodeDecodeError as e:
            logger.warning(f"Unicode decode error: {e}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected deserialization error: {e}")
            return None
    
    def process_message(self, message):
        """Process a single message from Kafka."""
        try:
            # Debug: log message metadata (topic/partition/offset) and length
            try:
                topic = getattr(message, 'topic', None)
                partition = getattr(message, 'partition', None)
                offset = getattr(message, 'offset', None)
                raw_value = getattr(message, 'value', None)
                val_len = len(raw_value) if raw_value is not None else 0
                logger.debug(f"Received message: topic={topic} partition={partition} offset={offset} value_len={val_len}")
                if logger.isEnabledFor(logging.DEBUG) and isinstance(raw_value, (bytes, bytearray)):
                    # show a short preview
                    preview = raw_value[:200]
                    try:
                        logger.debug(f"Raw preview: {preview.decode('utf-8', errors='replace')}")
                    except Exception:
                        logger.debug(f"Raw preview bytes: {preview}")
            except Exception:
                # Don't allow logging issues to stop processing
                pass

            tick = message.value
            
            # Skip if deserialization failed
            if tick is None:
                return None
            
            # Add tick to feature computer
            self.feature_computer.add_tick(tick)
            
            # Compute features
            features = self.feature_computer.compute_features(tick)
            
            # Publish to output topic
            self.producer.send(self.output_topic, value=features)
            
            # Add to batch for file writing
            self.features_batch.append(features)
            
            # Write batch to file periodically
            if len(self.features_batch) >= self.batch_size:
                self._write_batch()
            
            return features
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return None
    
    def _write_batch(self):
        """Write accumulated features to parquet file."""
        if not self.features_batch:
            return
        
        try:
            df = pd.DataFrame(self.features_batch)
            
            # Append to existing file or create new one
            if Path(self.output_file).exists():
                existing_df = pd.read_parquet(self.output_file)
                df = pd.concat([existing_df, df], ignore_index=True)
            
            df.to_parquet(self.output_file, index=False)
            logger.info(f"Wrote {len(self.features_batch)} features to {self.output_file}")
            
            self.features_batch = []
            
        except Exception as e:
            logger.error(f"Error writing batch: {e}")
    
    def run(self):
        """Run the feature pipeline."""
        logger.info("Starting feature pipeline...")
        message_count = 0
        processed_count = 0
        skipped_count = 0
        
        try:
            for message in self.consumer:
                message_count += 1
                
                features = self.process_message(message)
                
                if features:
                    processed_count += 1
                    if processed_count % 100 == 0:
                        logger.info(f"Processed {processed_count} messages (total seen: {message_count}, skipped: {skipped_count})")
                else:
                    skipped_count += 1
                    if skipped_count % 10 == 0:
                        logger.warning(f"Skipped {skipped_count} invalid messages so far")
                
        except KeyboardInterrupt:
            logger.info("Shutting down gracefully...")
        finally:
            # Write any remaining features
            self._write_batch()
            self.consumer.close()
            self.producer.close()
            logger.info(f"Pipeline stopped.")
            logger.info(f"  Total messages seen: {message_count}")
            logger.info(f"  Successfully processed: {processed_count}")
            logger.info(f"  Skipped (invalid): {skipped_count}")


def main():
    parser = argparse.ArgumentParser(description='Feature engineering pipeline')
    parser.add_argument('--topic_in', default='ticks.raw', help='Input Kafka topic')
    parser.add_argument('--topic_out', default='ticks.features', help='Output Kafka topic')
    parser.add_argument('--bootstrap_servers', default='localhost:9092', help='Kafka bootstrap servers')
    parser.add_argument('--output_file', default='data/processed/features.parquet', help='Output parquet file')
    parser.add_argument('--windows', nargs='+', type=int, default=[30, 60, 300], 
                        help='Window sizes in seconds')
    
    args = parser.parse_args()
    
    pipeline = FeaturePipeline(
        input_topic=args.topic_in,
        output_topic=args.topic_out,
        bootstrap_servers=args.bootstrap_servers,
        output_file=args.output_file,
        window_sizes=args.windows
    )
    
    pipeline.run()


if __name__ == '__main__':
    main()