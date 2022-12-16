using HDCtransform
using Plots

############ BASIC EXAMPLE 1: normalization issues
domain = collect(0:0.01:1);
nbins, lengthscale, D = 200,20, 100000
encoder = triangle_bin_encoder(nbins, lengthscale, D=D)#, normalization="theoretically")
f(x) = 1-x

## First approach: offset at boundaries
f_hdv = transform(f, encoder)
domain_hdv = encode(encoder, domain)
plot(domain, f.(domain))
plot!(domain, dot(f_hdv, domain_hdv))
vline!([lengthscale/nbins])

## Reason: function 'one': offset at boundaries: result of n^2(x) != n(x)n(x')
one(x)=1
one_hdv = transform(one, encoder)
plot(dot(one_hdv, encode(encoder, domain)))

## Second approach: correct with estimation of function one
plot(domain, f.(domain))
plot!(domain, dot(f_hdv, encode(encoder, domain)))
plot!(domain, dot(f_hdv, encode(encoder, domain)) ./ dot(one_hdv, encode(encoder, domain)))
vline!([lengthscale/nbins])
# Note: result okay, except for inevitable smoothing at boundaries

######### BASIC EXAMPLE 2: update n(x) in encoder such that function 'one' is correct (iterations, 10 by default)
encoder = triangle_bin_encoder(nbins, lengthscale, D=D)#, normalization="theoretically")
norms1 = [encoder.hdv_bins[i].norm for i in 1:length(encoder.hdv_bins)]
update(encoder, 10, true)
norms2 = [encoder.hdv_bins[i].norm for i in 1:length(encoder.hdv_bins)]
vline!([lengthscale])
# here, the function one is iteratively plotted and indeed approaches 1.
plot(norms1)
plot!(norms2)

######### Function one, in RD and -1/1^D
one_hdv = transform(one, encoder)
plot(domain, dot(one_hdv, encode(encoder, domain)), ylim=(0.5,1.5), label="real")
one_hdv_bipolar = tobipolar(one_hdv)
plot!(domain, dot(one_hdv_bipolar, encode(encoder, domain)), ylim=(0.5,1.5), label="bipolar")
vline!([encoder.lengthscale/encoder.nbins], label="lengthscale")


######### BASIC EXAMPLE 3: function estimate with updated n(x)
domain = collect(0:0.01:1);
nbins, lengthscale, D = 200,20, 100000
encoder = triangle_bin_encoder(nbins, lengthscale, D=D)#, normalization="theoretically")
f(x) = 1-x

plot(domain, f.(domain))
f_hdv = transform(f, encoder)
domain_hdv = encode(encoder, domain)
plot!(domain, dot(f_hdv, domain_hdv))
update(encoder)
f_hdv = transform(f, encoder)
domain_hdv = encode(encoder, domain)
plot!(domain, dot(f_hdv, domain_hdv))
vline!([lengthscale/nbins])

n2 = dot(f_hdv, f_hdv)
n2true = 1/3 # indeed <f,f> = int(f^2) \approx 1/3 
(n2-n2true)/n2true

######### EXAMPLE 4: updated encoder, different function
# definitions
domain = collect(0:0.01:1);
nbins, lengthscale, D = 500,25, 100000;
encoder = triangle_bin_encoder(nbins, lengthscale, D=D);#, normalization="theoretically")
update(encoder);
f(x) = x*sin(10x);

# transform, evaluate, and plot
f_hdv = transform(f, encoder);
domain_hdv = encode(encoder, domain);
plot(domain, f.(domain), legend=:topleft, label="true function")
plot!(domain, dot(f_hdv, domain_hdv), label="estimation")
vline!([1-lengthscale/nbins], label="1 - lengthscale")

# n2 norm 
n2 = dot(f_hdv, f_hdv)
n2true = 0.142 #wolfram alpha: 0.142
(n2-n2true)/n2true

# with bipolar vector
f_hdv_bipolar = tobipolar(f_hdv, encoder)
plot!(domain, dot(f_hdv_bipolar, encode(encoder, domain)), label="bipolar estimation")
dot(f_hdv_bipolar, f_hdv_bipolar)
#wolfram alpha: 0.142



############### EXAMPLE 5: data sample 
# definitions
domain = collect(0:0.01:1);
nbins, lengthscale, D = 500,40, 100000
encoder = triangle_bin_encoder(nbins, lengthscale, D=D)#, normalization="theoretically")
update(encoder)
f(x) = x*sin(10x)

# sample
x = rand(50); #collect(0:0.01:1)
fx = f.(x);

# transform and evaluate
f_hdv = transform(x, fx, encoder);
plot(x, fx, seriestype=:scatter, c=:red, label="",legend=:topleft)
plot!(domain, f.(domain), c=:red, label="true function")
plot!(x, dot(f_hdv, encode(encoder,x)), seriestype=:scatter, label="", c=:blue)
plot!(domain, dot(f_hdv, encode(encoder, domain)), label="estimation", c=:blue)

# n2 norm 
n2 = dot(f_hdv, f_hdv)
n2true = 0.142 #wolfram alpha: 0.142
(n2-n2true)/n2true

# with bipolar vector
f_hdv_bipolar = tobipolar(f_hdv, encoder)
plot!(domain, dot(f_hdv_bipolar, encode(encoder, domain)), label="bipolar estimation")
dot(f_hdv_bipolar, f_hdv_bipolar)
#wolfram alpha: 0.142



############### EXAMPLE 6: spectra: distributions with very short length scale 
# definitions
domain = collect(0:0.01:1);
nbins, lengthscale, D = 500,5, 100000
encoder = triangle_bin_encoder(nbins, lengthscale, D=D)#, normalization="theoretically")
update(encoder)
domain_hdv = encode(encoder, domain)

# artificial spectra: random points with hights
x = [0.1, 0.1417997608386008, 0.5, 0.6, 0.7, 0.75, 0.9]; 
fx = rand(7);
f_hdv = transform(x, fx, encoder);
plot(x, fx, seriestype=:scatter)
plot!(domain, dot(f_hdv, domain_hdv))
# note: peaks are approximated with finite widths with an area
# note: realistic data should also have a finite with, not a sum of dirac delta distributiosn
# note: if number round enough for bins: point reached. If number not round enough e.g. 0.1417997608386008, point not reached and instead peak a bit broader.

# normalization of a spectrum:
one(x) = 1;
one_hdv = transform(one, encoder);
dot(f_hdv, one_hdv)
# the latter dot product the area under the peaks: where peaks with triangles with lengthscale broad
sum(lengthscale/nbins * fx)


############### EXAMPLE 7: distributions of which only sample available
import Distributions
domain = collect(0:0.01:1);
nbins, lengthscale, D = 500,40, 100000
encoder = triangle_bin_encoder(nbins, lengthscale, D=D)#, normalization="theoretically")
update(encoder)
domain_hdv = encode(encoder, domain)

n=50
domain = collect(0:0.01:1)
domain_hdv = encode(encoder, domain)
p1 = Distributions.Normal(0.2, 0.05) 
p2 = Distributions.Normal(0.8, 0.05)
p3 = Distributions.Normal(0.6, 0.1)
p4 = Distributions.Normal(0.4, 0.07)

# sample x form the sum of p1 to p4
x = vcat(rand(p1, n), rand(p2,n), rand(p3,n), rand(p4,n)) #collect(0:0.01:1)
# p as pdf of sum p1 to p4 evaluated at domain
pdomain = (Distributions.pdf.(p1, domain).+Distributions.pdf.(p2, domain).+Distributions.pdf.(p3, domain).+.+Distributions.pdf.(p4, domain))/4

p_hdv = transform(x, encoder);
histogram(x, nbins=100, label="sample")
plot!(domain, 5*pdomain, c=:red, label="true density")
plot!(domain, 5*dot(p_hdv, domain_hdv), label="estimation", c=:blue)
vline!([lengthscale/nbins], label="lengthscale")

# area below estimated p_hdv indeed close to 1
one(x) = 1
one_hdv = transform(one, encoder)
dot(p_hdv, one_hdv)

f(x) = -x*sin(10x)
f_hdv = transform(f, encoder)

import LinearAlgebra
# compute expected value of f under p
# true estimation via function evaluation and density evaluation at each domain value
est1 = LinearAlgebra.dot(f.(domain), pdomain)/length(domain) 
# via dot product of estimations in hdv space
est2 = dot(f_hdv, p_hdv)
(est2-est1)/est1

f_hdv_bipolar = tobipolar(f_hdv, encoder)
p_hdv_bipolar = tobipolar(p_hdv, encoder)
est3 = dot(f_hdv_bipolar, p_hdv_bipolar)
(est3-est1)/est1


dot(f_hdv,f_hdv)
dot(f_hdv_bipolar, f_hdv_bipolar)


########### NOTE: influence of update on encoding and decoding


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
nbins, lengthscale = 200,50
D = 100000
domain = collect(0:0.01:1);
encoder1 = triangle_bin_encoder(nbins, lengthscale, D=D, boundary=false)
norms1 = [encoder1.hdv_bins[i].norm for i in 1:length(encoder1.hdv_bins)]
checkrecovery(encoder1, domain)
update(encoder1, 10)
norms2 = [encoder1.hdv_bins[i].norm for i in 1:length(encoder1.hdv_bins)]
checkrecovery(encoder1, domain)
vline!([lengthscale/nbins], label="lengthscale")
# note that due to the normalization correction, if one gets close to the boundary,
# then the maximimum likelihood estimate is set to the boundary. This is due to the 
# updated norms, i.e. norms2 here. At the boundaries, the normalizations become small and the likelyhood
# is increased.
plot!(collect(1:length(norms1))/length(norms1), norms1/maximum(norms1), label="norms1")
plot!(collect(1:length(norms1))/length(norms1), norms2/maximum(norms1), label="norms2")
#vline!([0.075])
# last note explanation: dirac delta spikes have a bias near the boundaries such that they are more equally represented:

domain_hdv = encode(encoder1, domain)
# by this dot product, each domain value is considered as a function (dirac delta) that is evaluated, and is the the distributed probablity on the domain values
P = dot(domain_hdv, domain_hdv)
pl = plot(domain, P[1,:], label="")
for i in 1:26
    plot!(domain, P[i,:], label="")
end
vline!([lengthscale/nbins], label="lengthscale")

one(x)=1
one_hdv = transform(one, encoder1)
areas = dot(one_hdv, domain_hdv)
plot(areas, ylims=(0.5,1.5))
# last remark: one can always do a MLE estimate neglecting the norms for decoding



##### for the boundary=true version: kind of non-boundary where function just extrapolates to zero at boundary
## recommended when: reasonable that function extrapolates near zero at boundaries, e.g. a data sample, 
## and when one is interested in recovery of the inputs
## here, the update function is not to be applied, n(x) is constant in the interval
############ BASIC EXAMPLE 1: revisited with boundary=true, here not recommended
domain = collect(0:0.01:1);
nbins, lengthscale, D = 200,20, 100000
encoder = triangle_bin_encoder(nbins, lengthscale, D=D, boundary=true)#, normalization="theoretically")
f(x) = 1-x

checkrecovery(encoder, domain)
vline!([lengthscale/nbins], label="lengthscale")

## First approach: offset at boundaries
f_hdv = transform(f, encoder)
domain_hdv = encode(encoder, domain)
plot(domain, f.(domain))
plot!(domain, dot(f_hdv, domain_hdv))
vline!([lengthscale/nbins], label="lengthscale")
dot(f_hdv, f_hdv)
# indeed the norm of f is a bit too small compared to 1/3

############### EXAMPLE 7: revisited with boundary=true, here possibily recommended: reasonable that density
# approaches zero at boundaries, and if one is interested
# in e.g. where is the probability maximal, and thus retrieving the input values
import Distributions
domain = collect(0:0.01:1);
nbins, lengthscale, D = 500,40, 100000
encoder = triangle_bin_encoder(nbins, lengthscale, D=D, boundary=true)#, normalization="theoretically")
domain_hdv = encode(encoder, domain)

n=50
domain = collect(0:0.01:1)
domain_hdv = encode(encoder, domain)
p1 = Distributions.Normal(0.2, 0.05) 
p2 = Distributions.Normal(0.8, 0.05)
p3 = Distributions.Normal(0.6, 0.1)
p4 = Distributions.Normal(0.4, 0.07)

# sample x form the sum of p1 to p4
x = vcat(rand(p1, n), rand(p2,n), rand(p3,n), rand(p4,n)) #collect(0:0.01:1)
# p as pdf of sum p1 to p4 evaluated at domain
pdomain = (Distributions.pdf.(p1, domain).+Distributions.pdf.(p2, domain).+Distributions.pdf.(p3, domain).+.+Distributions.pdf.(p4, domain))/4

p_hdv = transform(x, encoder)
m=5
histogram(x, nbins=100, label="sample")
plot!(domain, m*pdomain, c=:red, label="true density")
plot!(domain, m*dot(p_hdv, domain_hdv), label="estimation", c=:blue)
#vline!([1-lengthscale/nbins])

# area below estimated p_hdv indeed close to 1
one(x) = 1
one_hdv = transform(one, encoder)
dot(p_hdv, one_hdv)