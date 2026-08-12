// Accepts values produced by CounterProducer.
//
// The consumer is ready whenever enable is high.
// When valid data arrives while the consumer is ready,
// the value is stored in last_received_data.
module DataConsumer (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       enable,
    input  wire [7:0] data_in,
    input  wire       data_valid,
    output wire       ready_out
);

    reg [7:0] last_received_data;

    assign ready_out = enable;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            last_received_data <= 8'd0;
        end
        else begin
            if (data_valid && ready_out) begin
                last_received_data <= data_in;
            end
        end
    end

endmodule
