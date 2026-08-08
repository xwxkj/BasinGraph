# Track B development-only smoke protocol

The smoke gate uses official noiseless BBOB functions 1, 6, 10, 15 and 20;
dimensions 5 and 20; actual instance 1; all eight Track B algorithms; and a
budget of `100d`.

Expected runs:

```text
5 functions × 2 dimensions × 1 instance × 8 algorithms = 80
```

The smoke verifies installation, exact budget accounting, deterministic seed
mapping, observer output, baseline execution, result packaging and official
cocopp post-processing. It is engineering evidence only and cannot be pooled
with instances 21–30.
