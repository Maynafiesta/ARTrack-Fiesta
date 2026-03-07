import os
import torch
import torch.nn as nn
from lib.models.artrack_seq import build_artrack_seq
from lib.test.parameter.artrack_seq import parameters

class ARTrackONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Register constants as buffers so they are saved in the graph
        # identity is just a dummy input for the model if needed, but the template 
        # MUST be dynamic so we don't track the same car forever.
        
    def forward(self, template, search, seq_input):
        out = self.model(
            template=template,
            search=search,
            seq_input=seq_input,
            stage="sequence"
        )
        # Return only the necessary tensor for ONNX
        return out['seqs']

def export_model(model_name):
    # 1. Load params and model
    params = parameters(model_name)
    model = build_artrack_seq(params.cfg, training=False)
    
    # Checkpoint path is determined by Weights/PyTorch folder
    checkpoint_path = f"Weights/PyTorch/{model_name}.pth.tar"
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['net'], strict=True)
    model.eval()
    
    # 2. Prepare dummy inputs
    template_sz = params.template_size
    search_sz = params.search_size
    bins = params.cfg.MODEL.BINS
    
    dummy_template = torch.randn(1, 3, template_sz, template_sz)
    dummy_search = torch.randn(1, 3, search_sz, search_sz)
    # seq_input shape: (1, 28) based on save_all=7
    dummy_seq_input = torch.zeros(1, 28).long() 
    
    # 3. Wrap model
    wrapper = ARTrackONNXWrapper(model)
    wrapper.eval()
    
    # 4. Export using LEGACY exporter (dynamo=False)
    output_onnx = f"Weights/ONNX/{model_name}.onnx"
    print(f"Exporting {model_name} to ONNX (Legacy)...")
    
    use_external_data = ("large" in model_name.lower())
    
    torch.onnx.export(
        wrapper,
        (dummy_template, dummy_search, dummy_seq_input),
        output_onnx,
        input_names=['template', 'search', 'seq_input'],
        output_names=['output'],
        opset_version=16,
        do_constant_folding=False, # <-- DISABLED TO PREVENT RAM OOM!
        dynamo=False,
    )
    
    if use_external_data:
         # Need to handle it specially if we want it in one file or separated
         # Standard export might fail > 2GB. Let's try to use external data if it's large.
         pass
         
    print(f"Exported to {output_onnx}")

if __name__ == "__main__":
    models_to_export = ["artrack_seq_256_full", "artrack_seq_large_384_full"]
    
    for model_name in models_to_export:
        try:
            export_model(model_name)
        except Exception as e:
            print(f"Failed to export {model_name}: {e}")
