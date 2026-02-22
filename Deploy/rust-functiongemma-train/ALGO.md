graph TD
    %% Subgraph for High-Level Memory Context
    subgraph VRAM_Allocated ["GPU VRAM (RTX 5060 Ti - 16GB)"]
        direction TB
        
        %% Data Input
        Input[("Input Tokens<br>(Batch Size 1)")] --> Embeddings
        
        %% Model Layer Structure
        subgraph Transformer_Layer ["Transformer Block (Repeated N times)"]
            direction TB
            
            %% The Frozen Base Path (Compressed)
            subgraph Base_Path ["Frozen Base Model (Storage: ~4GB)"]
                NF4_Weights[("NF4 Quantized Weights<br>(4-bit Storage)")]
                LUT[("NF4 Lookup Table<br>(Constant)")]
                
                NF4_Weights -- "Index Lookup" --> Dequant_Kernel{{"Dequantization Kernel<br>(On-the-fly to BF16)"}}
                LUT -.-> Dequant_Kernel
            end
            
            %% The Trainable Adapter Path
            subgraph LoRA_Path ["Trainable Adapters (Storage: ~100MB)"]
                LoRA_A[("LoRA A<br>(BF16 / Rank r)")]
                LoRA_B[("LoRA B<br>(BF16 / Rank r)")]
            end
            
            %% Forward Pass Computation
            Dequant_Kernel --> Base_MatMul["Base MatMul (BF16)"]
            LoRA_A --> LoRA_MatMul["Adapter MatMul"] --> LoRA_B
            
            %% Attention Mechanism
            Base_MatMul & LoRA_B --> Sum_Out((+))
            Sum_Out --> FlashAttn{{"Flash Attention v2<br>(Memory Efficient)"}}
        end

        %% Gradient Checkpointing Logic
        subgraph Optimization_Logic ["Memory Optimization"]
            GC_Store["Gradient Checkpoint<br>(Store ONLY Input)"]
            GC_Discard["Discard Intermediate<br>Activations"]
        end
        
        FlashAttn --> GC_Store --> GC_Discard --> Output
    end
    
    %% Backward Pass & Updates
    Output --> Loss_Calc["Loss Calculation"]
    Loss_Calc == "Backward Pass" ==> Recompute
    
    subgraph Backprop ["Backward Pass (Compute Heavy)"]
        Recompute["Recompute Activations<br>(Re-run Forward from Checkpoint)"]
        Calc_Grads["Calculate Gradients"]
    end
    
    Recompute --> Calc_Grads
    Calc_Grads -- "Update Gradients" --> Optimizer
    
    subgraph Optimizer_State ["Optimizer (Host/GPU)"]
        Optim_Step["Optimizer Step<br>(AdamW/SGD)"]
    end
    
    Optimizer -- "Update Weights" --> LoRA_A & LoRA_B
    Optimizer -. "NO UPDATE" .-> NF4_Weights

    %% Styling
    style NF4_Weights fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    style LoRA_A fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style LoRA_B fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style Dequant_Kernel fill:#fff3e0,stroke:#ef6c00,stroke-dasharray: 5 5
    style FlashAttn fill:#fff3e0,stroke:#ef6c00,stroke-dasharray: 5 5
    style Recompute fill:#fce4ec,stroke:#c2185b,stroke-width:2px
