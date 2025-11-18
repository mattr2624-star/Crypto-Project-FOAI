from crypto-producer.producer import CryptoProducer
from crypto-consumer.consumer import CryptoConsumer

def run_stream(topic, bootstrap_servers):
    producer = CryptoProducer(bootstrap_servers)
    consumer = CryptoConsumer(topic, bootstrap_servers)

    # Example: sending and consuming messages
    producer.send(topic, "Hello Crypto Stream!")
    consumer.consume()
