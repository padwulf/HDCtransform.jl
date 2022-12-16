import Statistics
import Distributions

function triangle_bin_normalization(x, l)
    if x<l
        return l - (l-x)^2/l/2
    elseif x>1-l
        return l - (l-(1-x))^2/l/2
    else
        return l
    end
end
struct BIN_ENCODER
    hdv_bins
    lengthscale
    nbins
    boundary
end
# TODO: bin encoder more efficiently using bit vectors directly
function triangle_bin_encoder(nbins, lengthscale; D=10000, normalization=nothing, boundary=false)
    nodes = collect(1:lengthscale:nbins+lengthscale)
    bins = collect(nodes[1]:nodes[end])
    bins_hdv = zeros(length(bins),D)
    nodes_hdv = rand((1,-1), (length(nodes),D)) #hdv(length(nodes),D=D)
    samp = zeros(lengthscale+1, D)
    for i in 2:size(samp,1)
        j=i-1
        p_desired = j/lengthscale
        p = Statistics.mean(samp[i,:])
        s = Distributions.sample(findall(samp[i-1,:].==0),Int64.(round((p_desired-p)*D)), replace=false)
        samp[i:end, s].=1
    end
    for i in 1:length(nodes)-1
        for j in 1:lengthscale+1
            bins_hdv[nodes[i]+j-1,samp[j,:].==0] .= nodes_hdv[i  ,samp[j,:].==0]
            bins_hdv[nodes[i]+j-1,samp[j,:].==1] .= nodes_hdv[i+1,samp[j,:].==1]
        end
    end
    l = 1/nbins*lengthscale
    if normalization=="theoretically"
        n = sqrt.(D*[triangle_bin_normalization(i/nbins, l) for i in 1:nbins])
    elseif isnothing(normalization)
        #the inherent lengthscale l is lengthscale/nbins, thus divide by nbins
        #n = sqrt.(1/nbins*sum(bins_hdv*bins_hdv', dims=2)[:])
        # if number of bits really high, might consider this implementation:
        n = sqrt.(1/nbins*(sum(bins_hdv, dims=1)*bins_hdv')[:]) #not squared with n, but approximated by embedding of function 1.
    else
        n=0
    end
    hdv_bins = Vector{HDV}(undef,nbins)
    for i in 1:length(hdv_bins)
        hdv_bins[i] = hdv(bins_hdv[i,:].<0, n[i])
    end
    return BIN_ENCODER(hdv_bins, lengthscale, nbins, boundary)
end
function encode(encoder::BIN_ENCODER, x::Vector{Float64})
    n = encoder.nbins
    l = encoder.lengthscale
    if encoder.boundary
        xbin = Int64.(round.((n-1)*(x .+ l/n) ./ (1 + 2*l/n))).+1 #correction such that lengthscale l/n broader than boundary
    else
        xbin = Int64.(round.((n-1)*(x))).+1
    end
    x_hdv = Vector{BipolarHDV}(undef,length(x))
    for i in 1:size(x_hdv,1)
        x_hdv[i] = encoder.hdv_bins[xbin[i]]
    end
    return x_hdv
end
function decode(encoder::BIN_ENCODER, x_hdv::Vector{T}; confidence=nothing) where T<: HDV
    n = encoder.nbins
    l = encoder.lengthscale
    P = dot(x_hdv, encoder.hdv_bins)

    if encoder.boundary
        values = (collect(1:encoder.nbins).-1) /(encoder.nbins-1)*(1 + 2*l/n) .- l/n #correction such that lengthscale l/n broader than boundary
    else
        values = (collect(1:encoder.nbins).-1) /(encoder.nbins-1) 
    end
    MLE = [values[argmax(P[i,:])] for i in 1:size(P,1)]
    P = cancelnoise(P)
    #P = P ./ sum(P, dims=2) # normalize the probabilities
    P = (P .+ 1e-10) ./ (sum(P.+1e-10, dims=2)) #in case of cancelnoise and only noise, all probabilities are set to 0
    EVE = P*values
    if isnothing(confidence)
        return P, MLE, EVE
    else
        l,u = upperlowerbound(P, values, 1-confidence)
        return P, MLE, EVE, l,u
    end
end
function cancelnoise(P)
    noise = abs(minimum([minimum(P), 0]))   #size of noise is determined by maximum negative value. If no negative values, size of noise is 0.
    P[abs.(P).<= noise] .= 0  #noise is set to zero. Cancelling noise, especially for negative values, allows better normalization. 
    return P
end
function upperlowerbound(P, values, α)
    l = []
    u = []
    C = cumsum(P, dims=2)
    #println(C[1,:])
    C = (C.>α/2) .* (C.< 1-α/2)  
    for i in 1:size(P,1)
        v = values[C[i,:].==1]
        if length(v)==0  #can still throw error, if C switches from 0 to 1 in one step. Very narrow prediction, not interval. In this case, should increase lenghtscale. Or increase D dimensionality.
            push!(l, values[argmax(P[i,:])])
            push!(u, values[argmax(P[i,:])])
        else
            push!(l, v[1])  
            push!(u, v[end])
        end
    end
    return l,u
end
using Plots
function update(encoder, iterations=10, p=false)
    @assert encoder.boundary==false
    one(x)=1
    one_hdv = transform(one, encoder)
    one_ = dot(one_hdv, encoder.hdv_bins)
    if p  
        plot(one_, legend=:bottomleft)
    end
    for j in 1:iterations
        for i in 1:length(encoder.hdv_bins)
            encoder.hdv_bins[i].norm = encoder.hdv_bins[i].norm*sqrt(one_[i])
        end
        one_hdv = transform(one, encoder)
        one_ = dot(one_hdv, encoder.hdv_bins)
        if p
            plot!(one_)
        end
    end
end