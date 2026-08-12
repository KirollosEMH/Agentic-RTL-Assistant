// Produces incrementing 8-bit values.
//
// When:
//   1. enable is high, and
//   2. the downstream consumer is ready,
// the module increments data_out and marks it as valid for that cycle.
module CounterProducer (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       enable,
    input  wire       ready_in,
    output reg  [7:0] data_out,
    output reg        data_valid
);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_out   <= 8'd0;
            data_valid <= 1'b0;
        end
        else begin
            if (enable && ready_in) begin
                data_out   <= data_out + 8'd1;
                data_valid <= 1'b1;
            end
            else begin
                data_valid <= 1'b0;
            end
        end
    end

endmodule
