from cellpose import models
import inspect

# show model_path function source to understand download mechanism
print("=== model_path source ===")
print(inspect.getsource(models.model_path))

print()
print("=== cache_CPSAM_model_path source ===")
print(inspect.getsource(models.cache_CPSAM_model_path))
