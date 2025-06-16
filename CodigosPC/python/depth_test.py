import onnx

model = onnx.load("./models/depth_anything_v2_vits.onnx")
domains = set()
for node in model.graph.node:
    domains.add(node.domain)
    if node.domain and node.domain != "ai.onnx":
        print(f"Custom op: {node.op_type} | domain: {node.domain}")
print("Dominios únicos:", domains)