function transform(f, encoder)
    #@assert encoder.boundary==false #otherwise extrapolation to zero at boundaries
    n = encoder.nbins
    l = encoder.lengthscale
    res = zeros(size(encoder.hdv_bins[1].vector))
    if encoder.boundary
        values = (collect(1:encoder.nbins).-1) /(encoder.nbins-1)*(1 + 2*l/n) .- l/n #correction such that lengthscale l/n broader than boundary
    else
        values = (collect(1:encoder.nbins).-1) /(encoder.nbins-1) 
    end
    for i in 1:length(values)
        if (values[i]>0) & (values[i]<1) #this is for if the boundary is true, and also bins a little outside the interval are present
            res.+= f(values[i]) * encoder.hdv_bins[i].vector ./ encoder.hdv_bins[i].norm
            res.+= f(values[i]) * (encoder.hdv_bins[i].vector .-1) ./ encoder.hdv_bins[i].norm  #biploar vs bit
        end
    end
    return hdv(-res / length(values))
end
function transform(x::Vector{Float64},fx::Vector{Float64}, encoder)
    #@assert encoder.boundary==false #otherwise extrapolation to zero at boundaries
    res = zeros(size(encoder.hdv_bins[1].vector))
    x_hdv = encode(encoder, x)
    # TODO: this computation of nssq, should be done more efficiently
    A = zeros(length(x_hdv), length(x_hdv[1].vector))
    for i in 1:length(x_hdv)
        A[i,x_hdv[i].vector.==false].=1
        A[i,x_hdv[i].vector.==true].=-1
    end
    nssq = (sum(A, dims=1)*A')[:]
    for i in 1:length(x)
        res.+= fx[i] / nssq[i] * x_hdv[i].vector  * x_hdv[i].norm 
        res.+= fx[i] / nssq[i] * (x_hdv[i].vector .-1)  * x_hdv[i].norm
    end
    res=hdv(-res)
    return res
end
function ns(x_hdv)
    res = zeros(length(x_hdv[1].vector))
    for i in 1:length(x_hdv)
        res.+= x_hdv[i].vector
        res.+= (x_hdv[i].vector .-1)
    end
    dot(hdv(-res), [hdv(x_hdv[i].vector, 1) for i in 1:length(x_hdv)])
end
function transform(x_hdv::Vector{BipolarHDV},fx::Vector{Float64}, equallyspaced=false)
    #return 1
    res = zeros(length(x_hdv[1].vector))
    # TODO: this computation of nssq, should be done more efficiently
    #A = zeros(length(x_hdv), length(x_hdv[1].vector))
    #for i in 1:length(x_hdv)
    #    A[i,x_hdv[i].vector.==false].=1
    #    A[i,x_hdv[i].vector.==true].=-1
    #end
    #nssq = (sum(A, dims=1)*A')[:]
    if equallyspaced
        for i in 1:length(x_hdv)
            res.+= fx[i]  * x_hdv[i].vector  / x_hdv[i].norm 
            res.+= fx[i]  * (x_hdv[i].vector .-1)  / x_hdv[i].norm
        end
        res=hdv(-res / length(x_hdv))
    else
        nssq = ns(x_hdv) #sq stands for squared
        for i in 1:length(x_hdv)
            res.+= fx[i] / nssq[i] * x_hdv[i].vector  * x_hdv[i].norm 
            res.+= fx[i] / nssq[i] * (x_hdv[i].vector .-1)  * x_hdv[i].norm
        end
        res=hdv(-res)
    end
    return res
end

function transform(x::Vector{Float64}, encoder)
    #@assert encoder.boundary==false  #when sample, function may extrapolate to zero for out of sample
    res = zeros(size(encoder.hdv_bins[1].vector))
    x_hdv = encode(encoder, x)
    for i in 1:length(x)
        res.+= x_hdv[i].vector  / x_hdv[i].norm  #/ nssq[i]
        res.+= (x_hdv[i].vector .-1)  / x_hdv[i].norm #/ nssq[i]
    end
    res=hdv(-res / length(x))
    return res
end
function transform(x_hdv::Vector{RealHDV})
    res = zeros(length(x_hdv[1].vector))
    for i in 1:length(x_hdv)
        res.+= x_hdv[i].vector  # real values, thus not with the -1 or the norm contributions
    end
    res=hdv(res / length(x_hdv))  # real: also without minus sign #also:
    return res
end
function transform(x_hdv::Vector{BipolarHDV})
    res = zeros(length(x_hdv[1].vector))
    for i in 1:length(x_hdv)
        res.+= x_hdv[i].vector  / x_hdv[i].norm
        res.+= (x_hdv[i].vector .-1) / x_hdv[i].norm
    end
    res=hdv(-res / length(x))
    return res
end