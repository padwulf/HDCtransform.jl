using HDCtransform
using Base.Iterators
#using Plots

function rotate(encoder::BIN_ENCODER, shift)
    # create the hdv bins by circshift and use same norm, circhift should create new shifted, and not shift the old, see before
    newbins = [hdv(circshift(encoder.hdv_bins[i].vector,shift), encoder.hdv_bins[i].norm) for i in 1:length(encoder.hdv_bins)]
    return BIN_ENCODER(newbins, encoder.lengthscale, encoder.nbins, encoder.boundary)
end

struct MULTIVARIATE_ENCODER
    encoders::Vector
end
function euclidean_triangle_bin_encoder(n, nbins, lengthscale; D=10000, normalization=nothing, boundary=false)
    bin_encoder = triangle_bin_encoder(nbins, lengthscale; D=D, normalization, boundary)
    update(bin_encoder)
    encoders = [bin_encoder]
    for i in 2:n
        push!(encoders, rotate(bin_encoder,i-1))
    end
    return MULTIVARIATE_ENCODER(encoders)
end
function encode(encoder::MULTIVARIATE_ENCODER, X::Matrix{Float64})
    X_hdv = encode(encoder.encoders[1], X[:,1])
    for i in 2:size(X,2)
        X_hdv = X_hdv .⊗ encode(encoder.encoders[i], X[:,i])
    end
    return X_hdv
end
#function ns(x_hdv)
#    res = zeros(length(x_hdv[1].vector))
#    for i in 1:length(x_hdv)
#        res.+= x_hdv[i].vector
#        res.+= (x_hdv[i].vector .-1)
#    end
#    dot(hdv(-res), [hdv(x_hdv[i].vector, 1) for i in 1:length(x_hdv)])
#end
function transform(f, encoder::MULTIVARIATE_ENCODER)
    @assert length(encoder.encoders)==2 # not yet generally for every number of dimensions
    r = 0:0.01:1 #stepsize for integral in all dimensions equal
    x_grid = collect(product(r,r))

    #domain_grid = collect(product(r,r))
    #domain = collect(reduce(hcat, collect.(vec(domain_grid)))')

    x = collect(reduce(hcat, collect.(vec(x_grid)))')
    fx = [f(x[i,:]) for i in 1:size(x,1)]
    x_hdv = encode(encoder, x)
    res = zeros(length(x_hdv[1].vector))
    for i in 1:length(x_hdv)
        res.+= fx[i] / x_hdv[i].norm * x_hdv[i].vector    
        res.+= fx[i] / x_hdv[i].norm * (x_hdv[i].vector .-1)  
    end
    res=hdv(-res /length(x_hdv))
    return res

end
function transform(x::Matrix{Float64},fx::Vector{Float64}, encoder::MULTIVARIATE_ENCODER) where T<:HDV
    x_hdv = encode(encoder, x)
    res = zeros(length(x_hdv[1].vector))
    nssq = ns(x_hdv) #sq stands for squared
    for i in 1:length(x_hdv)
        res.+= fx[i] / nssq[i] * x_hdv[i].norm * x_hdv[i].vector    
        res.+= fx[i] / nssq[i] * x_hdv[i].norm * (x_hdv[i].vector .-1)  
    end
    res=hdv(-res)
    return res
end
function transform(x::Matrix{Float64}, encoder::MULTIVARIATE_ENCODER)
    res = zeros(size(encoder.encoders[1].hdv_bins[1].vector))
    x_hdv = encode(encoder, x)
    for i in 1:size(x,1)
        res.+= x_hdv[i].vector  / x_hdv[i].norm  #/ nssq[i]
        res.+= (x_hdv[i].vector .-1)  / x_hdv[i].norm #/ nssq[i]
    end
    res=hdv(-res / size(x,1))
    return res
end