// Top-level module.
// Connects CounterProducer directly to DataConsumer.
// A future task can insert a FIFO between these two modules.
module DataPipeline (
    input wire clk,
    input wire rst_n,
    input wire producer_enable,
    input wire consumer_enable
);

    wire [7:0] produced_data;
    wire       produced_data_valid;
    wire       consumer_ready;

    CounterProducer u_counter_producer (
        .clk        (clk),
        .rst_n      (rst_n),
        .enable     (producer_enable),
        .ready_in   (consumer_ready),
        .data_out   (produced_data),
        .data_valid (produced_data_valid)
    );

    DataConsumer u_data_consumer (
        .clk        (clk),
        .rst_n      (rst_n),
        .enable     (consumer_enable),
        .data_in    (produced_data),
        .data_valid (produced_data_valid),
        .ready_out  (consumer_ready)
    );

endmodule
