# BASICS
############################################"
import LinearAlgebra
import Base.sum
abstract type HDV
end

mutable struct BipolarHDV <: HDV
    vector::BitVector
    norm::Float64
end
struct RealHDV <: HDV
    vector::Vector{Float64}
end
function hdv(vector::Vector)
    return RealHDV(Float64.(vector))
end
function hdv(vector::BitVector, norm)
    return BipolarHDV(vector, Float64(norm))
end
function ⊗(a::BipolarHDV, b::BipolarHDV)
    return BipolarHDV(xor.(a.vector,b.vector), a.norm*b.norm/sqrt(length(a.vector))) #norm should decrease
end
function ⊗(a::RealHDV, b::RealHDV)
    return RealHDV(a.vector .* b.vector *sqrt(length(a.vector))) #similarly normalization somaller
end
function ⊗(a::BipolarHDV, b::RealHDV)
    v = b.vector / a.norm
    v[a.vector.==true] .*=-1
    return RealHDV(v *sqrt(length(a.vector)))
end
function ⊗(b::RealHDV, a::BipolarHDV,)
    return a⊗b
end
function ⊗(a::Vector{T}, b::Vector{T}) where T<: HDV
    res = Matrix{T}(undef,length(a), length(b))
    for i in 1:size(res,1)
        for j in 1:size(res,2)
            res[i,j] = a[i] ⊗ b[j]
        end
    end
    return res
end
function ⊗(a::RealHDV, b::Vector{T}) where T<: HDV
    res = Vector{RealHDV}(undef,length(b))
    for i in 1:length(res)
        res[i] = a ⊗ b[i]
    end
    return res
end
function ⊗(a::T, b::Vector{T}) where T<: HDV
    res = Vector{T}(undef,length(b))
    for i in 1:length(res)
        res[i] = a ⊗ b[i]
    end
    return res
end
function ⊗(b::Vector{T}, a::HDV) where T<: HDV
    return a⊗b
end



function dot(a::BipolarHDV, b::BipolarHDV)
    return (2*sum(a.vector.==b.vector) - length(a.vector)) / a.norm / b.norm
end
function dot(a::RealHDV, b::BipolarHDV)
    return (sum(a.vector[b.vector.==false]) - sum(a.vector[b.vector.==true])) / b.norm
end
function dot(a::BipolarHDV, b::RealHDV)
    return dot(b,a)
end
function dot(a::RealHDV, b::RealHDV)
    return LinearAlgebra.dot(a.vector,b.vector)
end
function dot(a::Vector{Ta}, b::Vector{Tb}) where {Ta<: HDV, Tb<:HDV}
    res = zeros(length(a), length(b))
    for i in 1:size(res,1)
        for j in 1:size(res,2)
            res[i,j] = dot(a[i],b[j])
        end
    end
    return res
end
function dot(a::HDV, b::Vector{T}) where T<: HDV
    res = zeros(length(b))
    for i in 1:length(res)
        res[i] = dot(a,b[i])
    end
    return res
end
function dot(b::Vector{T}, a::HDV) where T<: HDV
    return dot(a,b)
end
"""
function sum(a::Vector{HDV})
    res = zeros(length(a[1].vector))
    for i in 1:length(a)
        res+= a[i].vector
    end
    return res
end
"""
function tobipolar(realHDV::RealHDV, encoder)
    vector = realHDV.vector .< 0
    res = hdv(vector, 1)
    c = Statistics.mean(abs.(dot(res, encoder.hdv_bins) ./ dot(realHDV, encoder.hdv_bins)))
    #c = sqrt(length(vector)) / LinearAlgebra.norm(realHDV.vector)
    res.norm = c
    return res
end
function tobipolar(realHDV::RealHDV)
    vector = realHDV.vector .< 0
    res = hdv(vector, 1)
    #c = Statistics.mean(dot(res, encoder.hdv_bins) ./ dot(realHDV, encoder.hdv_bins))
    c = sqrt(length(vector)) / LinearAlgebra.norm(realHDV.vector)
    res.norm = c
    return res
end