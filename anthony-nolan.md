Running this on windows requires `nvcc` Nvidia's cuda compiler.
Also clang.

I broadcasted the positional embeddings across the x-batch rather than expand them explicitly to the same dim.