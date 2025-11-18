```mermaid
flowchart LR
    subgraph DataSource
        A[Coinbase BTC-USD Stream\n(or replayed CSV/NDJSON)]
    end

    subgraph KafkaLayer
        B[crypto-producer\n(Kafka producer)]
        C[(Kafka Broker)]
    end

    subgraph ModelServing
        D[FastAPI Model Server\n/health, /predict, /version, /metrics]
    end

    subgraph Monitoring
        E[Prometheus]
        F[Grafana]
        G[Drift Monitor\n(Evidently)]
    end

    A --> B --> C
    C --> D
    D -->|Prometheus metrics| E --> F
    D -->|Model events\n(retrain)| C
    C -->|feature data| G -->|drift reports| F
