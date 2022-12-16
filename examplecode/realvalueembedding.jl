using HDCtransform
using Plots

### Check boundary effects
# boundary argument false (by default): do not cover extra lengthscale at boundaries
# note if confidence not given, l and u limits are not returned
function checkrecovery(encoder, domain)
    domain_hdv = encode(encoder, domain)
    P, MLE, EVE, l, u = decode(encoder, domain_hdv, confidence=0.95)  #l and u, lower and upperbound only return if confidence is geven
    plot(domain, MLE, label="MLE",legend=:bottomright)
    plot!(domain, EVE, label="EVE")
    plot!(domain, Float64.(l), fillrange = Float64.(u), fillalpha = 0.1, c = 1, label = "95% prediction interval")
    plot!(domain, Float64.(u), fillrange = Float64.(l), fillalpha = 0.1, c = 1, label = nothing)
end
nbins, lengthscale = 200,40
D = 100000
domain = collect(0:0.01:1);
encoder1 = triangle_bin_encoder(nbins, lengthscale, D=D, boundary=false)
encoder2 = triangle_bin_encoder(nbins, lengthscale, D=D, boundary=true)
checkrecovery(encoder1, domain)
checkrecovery(encoder2, domain)



### Check difference between theoretical and experimental normalization (area below the triangles)
encoder = triangle_bin_encoder(nbins, lengthscale, D=D)
experimental_norms = [encoder.hdv_bins[i].norm for i in 1:length(encoder.hdv_bins)]
encoder = triangle_bin_encoder(nbins, lengthscale, D=D, normalization="theoretically")
theoretical_norms = [encoder.hdv_bins[i].norm for i in 1:length(encoder.hdv_bins)]
plot(theoretical_norms, label="theoretical_norms")
plot!(experimental_norms, label="experimental_norms")


### Check similarities (kernels)

# boundary false
encoder = triangle_bin_encoder(nbins, lengthscale, D=D)
domain
domain_hdv = encode(encoder, domain)
K = dot(domain_hdv, domain_hdv)
heatmap(K)
plot(K[50,:])
# note: at the boundaries, similarities to itself increase (normalization). (Doubles: half a triangle)

# boundary true
encoder = triangle_bin_encoder(nbins, lengthscale, D=D, boundary=true)
domain_hdv = encode(encoder, domain)
K = dot(domain_hdv, domain_hdv)
heatmap(K)
plot(K[50,:])
# note: at the boundaries, similarities to itself do not increase (still bins outside the interval, and similaritie to yourself increases there)