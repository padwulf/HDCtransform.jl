module HDCtransform

# Write your package code here.
export hdv, ⊗, dot, tobipolar
export triangle_bin_encoder, encode, decode, update
export transform
export euclidean_triangle_bin_encoder
export cancelnoise

#using Distributions

include("basichdv.jl")
include("realvalueembedding.jl")
include("transform.jl")
include("multivariate.jl")
#include("mult.jl")
end
