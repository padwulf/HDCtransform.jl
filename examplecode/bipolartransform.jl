using HDCtransform
using Plots


######### BASIC EXAMPLE 3: function estimate with updated n(x)
domain = collect(0:0.01:1);
nbins, lengthscale, D = 200,20, 100000
encoder = triangle_bin_encoder(nbins, lengthscale, D=D)#, normalization="theoretically")
update(encoder)
vline!([lengthscale/nbins])


f(x) = 1-x
f_hdv = transform(f, encoder)
domain_hdv = encode(encoder, domain)
plot(domain, f.(domain))
vline!([lengthscale/nbins])
plot!(domain, dot(f_hdv, domain_hdv))
plot!(domain, dot(tobipolar(f_hdv), domain_hdv))

import Statistics
########## integral under the joint points
one(x)=1
one_hdv = transform(one, encoder)
n=dot(one_hdv, encoder.hdv_bins)
sum(n)
plot(n, ylim=(0.5,1.5))
n=dot(tobipolar(one_hdv), encoder.hdv_bins)
plot(n, ylim=(0.5,1.5))
sum(n)
correction = dot(tobipolar(one_hdv), encoder.hdv_bins)
sum(n)/ Statistics.mean(correction)
plot(n./correction, ylim=(0.5,1.5))



#################################
f(x) = 1-x
f_hdv = transform(f, encoder)
domain_hdv = encode(encoder, domain)
plot(domain, f.(domain))
vline!([lengthscale/nbins])
plot!(domain, dot(f_hdv, domain_hdv))
plot!(domain, dot(tobipolar(f_hdv), domain_hdv))
one(x)=1
one_hdv = transform(one, encoder)
plot!(domain, dot(tobipolar(f_hdv), domain_hdv) ./ Statistics.mean(correction))