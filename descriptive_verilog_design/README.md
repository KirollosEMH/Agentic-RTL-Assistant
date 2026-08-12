# Simple Verilog Data Pipeline

This project is intentionally small and beginner-friendly.

## Modules

### `DataPipeline`
Top-level module. It creates and connects:

- `CounterProducer`
- `DataConsumer`

The producer and consumer are directly connected.

### `CounterProducer`
Produces incrementing 8-bit numbers.

It generates a new value only when:

- `enable` is high, and
- `ready_in` says the consumer can accept data.

### `DataConsumer`
Receives values from the producer.

It is ready whenever its `enable` input is high. When `data_valid` is high while
the consumer is ready, it stores the received value.

## Data flow

    CounterProducer  ------ data + valid ------>  DataConsumer
          ^                                         |
          |--------------- ready -------------------|

