using HDCtransform


#####################

using Base.Iterators
r = 0:0.01:1
axis = collect(r)
domain_grid = collect(product(r,r))
domain = collect(reduce(hcat, collect.(vec(domain_grid)))')

nbins,lengthscale,D = 200,20, 100000
encoder = euclidean_triangle_bin_encoder(2, nbins, lengthscale, D=D)
domain_hdv = encode(encoder, domain)

#### check if encoder are correct and independet
#1. rotated version results in same correlations, while is uncorrelated to orriginal
enc1 = encode(encoder.encoders[1], axis)
enc2 = encode(encoder.encoders[2], axis)
P1 = dot(enc1, enc1)[25,:]
P2 = dot(enc2, enc2)[25,:]
P12 = dot(enc1, enc2)[25,:]
plot(P1)
plot!(P2.+1e-1)
plot!(P12)
# Ok:
#2. test if outer product ⊗ is universal
ax = enc1[50]
ay = enc2[70]
bx = enc1[52]
by = enc2[72]
dot(ax, bx)
dot(ay, by)
r1 = dot(ax, bx)*dot(ay, by)
r2 = dot(ax ⊗ ay, bx ⊗ by)
# Ok
#3. test integral below dirac delta functions



### Transform: approximation

function f1(args::Vector)
    m1 = [0.35,0.35]
    m2 = [0.65,0.65]
    s = 0.15
    return exp(-LinearAlgebra.norm(((args .-m1) /s))^2) + exp(-LinearAlgebra.norm(((args .-m2) /s))^2)
end
f1_d = [f1(domain[i,:]) for i in 1:size(domain,1)]
heatmap(reshape(f1_d,size(domain_grid)))


# 1. function given, integral approximated
f1_hdv = transform(f1, encoder)
f1_recovered = dot(f1_hdv, domain_hdv)
plot(f1_d, f1_recovered, seriestype=:scatter)
heatmap(reshape(f1_recovered, size(domain_grid)))


f1_hdv_bipolar = tobipolar(f1_hdv)
f1_recovered = dot(f1_hdv_bipolar, domain_hdv)
plot(f1_d, f1_recovered, seriestype=:scatter)
heatmap(reshape(f1_recovered, size(domain_grid)))


# 2. sample given, approximated
x = rand(500,2)
fx = [f1(x[i,:]) for i in 1:size(x,1)]
scatter(x[:,1], x[:,2], c=cgrad()[fx])
f1_hdv = transform(x, fx, encoder);
f1_recovered = dot(f1_hdv, domain_hdv)
plot(f1_d, f1_recovered, seriestype=:scatter)
heatmap(reshape(f1_recovered, size(domain_grid)))

# 3. density sample
factor = 0.1
x = rand(10000,2)
fx = [f1(x[i,:]) for i in 1:size(x,1)]
px = factor*fx
x= x[px.>rand(size(x,1)),:]
scatter(x[:,1], x[:,2])
length(x)
f1_hdv = transform(x, encoder)
f1_recovered = dot(f1_hdv, domain_hdv)
plot(f1_d, f1_recovered, seriestype=:scatter)
heatmap(reshape(f1_recovered, size(domain_grid)))

### Transform: conditioning operations
f1_hdv = transform(f1, encoder);
condition = 0.55
condition_hdv = encode(encoder.encoders[1], [condition])[]
f1_conditional = f1_hdv ⊗ condition_hdv
plot(dot(f1_conditional, encode(encoder.encoders[2], axis)))
plot!([f1([condition, i]) for i in axis])

### Transform: integral operations
one(x)=1
one_hdv1 = transform(one, encoder.encoders[1])
one_hdv2 = transform(one, encoder.encoders[2])
f1_hdv = transform(f1, encoder);


# 1. integration of function transform
## 1.a. total integral
dot(f1_hdv, one_hdv1 ⊗ one_hdv2)
Statistics.mean(f1_d) #unit intervals, integral is mean value


## 1.b. conditional integral ~ marginal function in non integrated variables
conditions = axis
conditions_hdv = encode(encoder.encoders[1], conditions)
conditionalfunctions = f1_hdv ⊗ conditions_hdv
plot(dot(conditionalfunctions, one_hdv2))
plot!(Statistics.mean(reshape(f1_d, size(domain_grid)), dims=1)[:])

### 2. Integration of sample transform
factor = 0.1
x = rand(10000,2)
fx = [f1(x[i,:]) for i in 1:size(x,1)]
px = factor*fx
x= x[px.>rand(size(x,1)),:]
scatter(x[:,1], x[:,2])
length(x)
f1_hdv = transform(x, encoder)
dot(f1_hdv, one_hdv1 ⊗ one_hdv2)
# indeed, always tends to totoal integral of 1. Sample interpreted as distribution.


######### Check dirac deltas
# each domain encoded value should represent a dirac delta function with area below equals 1
domain
domain_hdv = encode(encoder,domain)
plot(dot(domain_hdv, one_hdv1 ⊗ one_hdv2),ylim=(0.5,1.5))
axis1_hdv = encode(encoder.encoders[1], axis)

# left: multiplication with one_hdv2 makes it a marginal function integrated over dim 2, then
# dot product with right results in marginal probabilities over dim 1 ~ decoding for 1: e.g. MLE, EVE
P = dot(domain_hdv[1:150] ⊗ one_hdv2, axis1_hdv)
heatmap(P', label="marginal probabilities")
plot!(length(axis)*domain[1:150,1], label="x value in domain")
# comment 1: these probabilites are used in the bin encoder for the decode function for MLE, EVE, etc
# comment 2: dot( a ⊗ b, c) = dot(a ⊗ c, b) # see interpretation for above



######### Check regression examples
using Distributions
### 1.
function dataset1(n)
    x = rand(n)
    noisedistribution = Normal(0,0.2)
    ytrue = x
    y = ytrue + ytrue.*rand(noisedistribution, n);
    yaxis = axis
    #standardscaling
    u, s =minimum(y), maximum(y)-minimum(y)
    y = (y.-u)/s
    ytrue = (ytrue.-u)/s
    yaxis = (yaxis.-u)/s
    return x,y,yaxis
end
function dataset2(n)
    x = rand(n)
    noisedistribution = Normal(0,0.05)
    ytrue = x.*((sin.(20*x) .+1) ./2)
    yaxis = axis.*((sin.(20*axis) .+1) ./2)
    y = ytrue + rand(noisedistribution, n);
    #standardscaling
    u, s =minimum(y), maximum(y)-minimum(y)
    y = (y.-u)/s
    ytrue = (ytrue.-u)/s
    yaxis = (yaxis.-u)/s
    return x,y,yaxis
end
function evaluate(x,y,yaxis, encoder)
    model_hdv = transform(hcat(x,y), encoder)
    # predictions are conditional function evaluations
    predictions = model_hdv ⊗ encode(encoder.encoders[1], axis)
    # predictions are decoded: into probabilities, MLE, EVE, l and u bound
    P, MLE, EVE, l, u = decode(encoder.encoders[2], predictions, confidence=0.95)  #l and u, lower and upperbound only return if confidence is geven
    #heatmap(P)
    plot(x,y, seriestype=:scatter, alpha=0.5, label="data", legend=:topleft)
    plot!(axis, MLE, label="MLE")
    plot!(axis, EVE, label="EVE")
    plot!(axis, Float64.(l), fillrange = Float64.(u), fillalpha = 0.1, c = 1, label = "95% prediction interval")
    plot!(axis, Float64.(u), fillrange = Float64.(l), fillalpha = 0.1, c = 1, label = nothing)
    plot!(axis, yaxis, c=:black, label="true values")
end


### Dataset 1
x,y,yaxis = dataset1(100)
plot(x,y,seriestype=:scatter, alpha=0.3)
plot!(axis,yaxis, c=:black)

# lengtshscael 1/20
encoder = euclidean_triangle_bin_encoder(2, 200, 15, D=100000)
evaluate(x,y,yaxis,encoder)
vline!([15/200], label="lengthscale")

# lengthscale 1/20 - D = 10000 
encoder = euclidean_triangle_bin_encoder(2, 200, 10, D=10000)
evaluate(x,y,yaxis,encoder)

# lengthscale 1/10
encoder = euclidean_triangle_bin_encoder(2, 200, 20, D=100000)
evaluate(x,y,yaxis,encoder)

# lengthscale 1/10 - D = 10000 
encoder = euclidean_triangle_bin_encoder(2, 200, 20, D=10000)
evaluate(x,y,yaxis,encoder)

### Dataset 2
x,y,yaxis = dataset2(100)
plot(x,y,seriestype=:scatter, alpha=0.3)
plot!(axis,yaxis, c=:black)

# lengtshscael 1/20
encoder = euclidean_triangle_bin_encoder(2, 200, 10, D=100000)
evaluate(x,y,yaxis,encoder)
vline!([1/20], label="lengthscale")

# lengthscale 1/10
encoder = euclidean_triangle_bin_encoder(2, 200, 20, D=100000)
evaluate(x,y,yaxis,encoder)
vline!([1/10], label="lengthscale")


### Dataset 2 - few data points
x,y,yaxis = dataset2(25)
plot(x,y,seriestype=:scatter, alpha=0.3)
plot!(axis,yaxis, c=:black)

# lengtshscael 1/20
encoder = euclidean_triangle_bin_encoder(2, 200, 15, D=100000)
evaluate(x,y,yaxis,encoder)
vline!([15/200], label="lengthscale")


# lengthscale 1/10
encoder = euclidean_triangle_bin_encoder(2, 200, 20, D=100000)
evaluate(x,y,yaxis,encoder)
vline!([1/10], label="lengthscale")


############# Add bipolar approx: very same is repeated
function evaluate_bipolar(x,y,yaxis, encoder)
    model_hdv = tobipolar(transform(hcat(x,y), encoder))
    # predictions are conditional function evaluations
    predictions = model_hdv ⊗ encode(encoder.encoders[1], axis)
    # predictions are decoded: into probabilities, MLE, EVE, l and u bound
    P, MLE, EVE, u, l = decode(encoder.encoders[2], predictions, confidence=0.95)  #l and u, lower and upperbound only return if confidence is geven
    #heatmap(P)
    plot(x,y, seriestype=:scatter, alpha=0.5, label="data", legend=:topleft)
    plot!(axis, MLE, label="MLE")
    plot!(axis, EVE, label="EVE")
    plot!(axis, Float64.(l), fillrange = Float64.(u), fillalpha = 0.1, c = 1, label = "95% prediction interval")
    plot!(axis, Float64.(u), fillrange = Float64.(l), fillalpha = 0.1, c = 1, label = nothing)
    plot!(axis, yaxis, c=:black, label="true values")
end

### Dataset 1
x,y,yaxis = dataset1(100)
plot(x,y,seriestype=:scatter, alpha=0.3)
plot!(axis,yaxis, c=:black)

# lengtshscael 1/20
encoder = euclidean_triangle_bin_encoder(2, 200, 10, D=100000)
evaluate_bipolar(x,y,yaxis,encoder)
# lengthscale 1/10
encoder = euclidean_triangle_bin_encoder(2, 200, 20, D=100000)
evaluate_bipolar(x,y,yaxis,encoder)



### Dataset 2
x,y,yaxis = dataset2(100)
plot(x,y,seriestype=:scatter, alpha=0.3)
plot!(axis,yaxis, c=:black)

# lengtshscael 1/20
encoder = euclidean_triangle_bin_encoder(2, 200, 10, D=100000)
evaluate_bipolar(x,y,yaxis,encoder)
vline!([1/20], label="lengthscale")

# lengthscale 1/10
encoder = euclidean_triangle_bin_encoder(2, 200, 20, D=100000)
evaluate_bipolar(x,y,yaxis,encoder)
vline!([1/10], label="lengthscale")


### Dataset 2 - few data points
x,y,yaxis = dataset2(50)
plot(x,y,seriestype=:scatter, alpha=0.3)
plot!(axis,yaxis, c=:black)

# lengtshscael 1/20
encoder = euclidean_triangle_bin_encoder(2, 200, 10, D=100000)
evaluate_bipolar(x,y,yaxis,encoder)
vline!([1/20], label="lengthscale")

# lengthscale 1/10
encoder = euclidean_triangle_bin_encoder(2, 200, 20, D=100000)
evaluate_bipolar(x,y,yaxis,encoder)
vline!([1/10], label="lengthscale")


########## Uncertainty prediction exponential, gaussian
function dataset_noise(n, noisedistribution)
    x = rand(n)
    # make sure unit interval (limiting easier than standardscaling to maintain original noise distribution)
    ytrue = x
    y = ytrue + rand(noisedistribution, n);
    x = x[(y.>0) .& (y.<1)]    
    y = y[(y.>0) .& (y.<1)]    
    return x,y
end
function evaluate_noise(x,y, encoder)
    model_hdv = transform(hcat(x,y), encoder)
    value = 0.5
    predictions = model_hdv ⊗ encode(encoder.encoders[1], [value])
    P = dot(encode(encoder.encoders[2], axis), predictions)
    P = cancelnoise(P)
    P = P ./ sum(P)
    plot(axis, P[:])
    pdf_ = pdf.(noisedistribution, axis.-0.5)
    plot!(axis, pdf_/sum(pdf_))
end

noisedistribution = Exponential(0.1)

x,y  = dataset_noise(200, noisedistribution)
plot(x,y,seriestype=:scatter, alpha=0.3)
plot!(axis, axis, c=:black)
encoder = euclidean_triangle_bin_encoder(2, 200, 10, D=100000)
evaluate_noise(x,y, encoder)


noisedistribution = Normal(0,0.05)

x,y  = dataset_noise(200, noisedistribution)
plot(x,y,seriestype=:scatter, alpha=0.3)
plot!(axis, axis, c=:black)
encoder = euclidean_triangle_bin_encoder(2, 200, 15, D=100000)
evaluate_noise(x,y, encoder)



### Data Daan.. not yet clear. Do synthetically
## define the data 
using Distributions
axis = collect(0:0.01:1)
n=400
σ1 = 0.08
σ2 = 0.06
var1 = rand(Uniform(3*σ1,1-3σ1), n);
var2 = rand(Uniform(3*σ2,1-3σ2), n);
A = collect(reduce(hcat,[pdf.(Normal(var1[i],σ1),axis) .+ pdf.(Normal(var2[i],σ2),axis) for i in 1:n])');
A= A ./ sum(A,dims=2);
s = sample(1:size(A,1), 3, replace=false)
p=plot(legend=nothing)
for si in s #1:n # s
    plot!(A[si,:])
end
p
var1
var2
function scale(x)
    return (x .- minimum(x)) /(maximum(x)-minimum(x)) 
end
var1 = scale(var1)
var2 = scale(var2)
traininds = collect(1:size(A,1))
deleteat!(traininds,sort(s))

## Method 1: encode the distributions and bind them with the variables
### 1.1 Encode distributins
encoder_dist = triangle_bin_encoder(200,10, D=100000)
axis_hdv = encode(encoder_dist, axis)
@time distributions_hdv = [transform(axis_hdv, A[i,:], true) for i in 1:size(A,1)];
p=plot()
for si in s
    d = dot(distributions_hdv[si], axis_hdv)
    d = cancelnoise(d)
    d = d/sum(d)
    plot!(d)
    plot!(A[si,:])
end
p


### 1.2 Encode the varibales, bind, transform, and predict
var_ = hcat(var1, var2)
encoder = euclidean_triangle_bin_encoder(2, 200,25,D=100000)
var_hdv = encode(encoder, var_)

model = transform(var_hdv[traininds] .⊗ distributions_hdv[traininds]) 

p=plot()
for si in s
    prediction = model ⊗ var_hdv[si]
    prediction_axis = dot(prediction, axis_hdv) #/ dot(prediction, one_hdv)
    prediction_axis = cancelnoise(prediction_axis)
    prediction_axis = prediction_axis/sum(prediction_axis)*sum(A[si,:])
    plot!(prediction_axis)
    plot!(A[si,:])
end
p

## Method 2: bind the variables with the domain variables, and transform with the values
x = var_hdv[traininds] ⊗ axis_hdv
y = A[traininds,:]

@time model = transform(vec(x), vec(y));

p=plot()
for si in s
    prediction = model ⊗ var_hdv[si]
    prediction_axis = dot(prediction, axis_hdv) #/ dot(prediction, one_hdv)
    prediction_axis = cancelnoise(prediction_axis)
    prediction_axis = prediction_axis/sum(prediction_axis)*sum(A[si,:])
    plot!(prediction_axis)
    plot!(A[si,:])
end
p

### Method 3: encode variables, encode distribution, and do linear regression between them: less dimensions required for stable solution
### 3.1 Encode distributins
encoder_dist = triangle_bin_encoder(200,10, D=10000)
axis_hdv = encode(encoder_dist, axis)
@time distributions_hdv = [transform(axis_hdv, A[i,:], true) for i in 1:size(A,1)];
p=plot()
for si in s
    d = dot(distributions_hdv[si], axis_hdv)
    d = cancelnoise(d)
    d = d/sum(d)
    plot!(d)
    plot!(A[si,:])
end
p

### 3.2 Encode the varibales, bind, transform, and predict
var_ = hcat(var1, var2)
encoder = euclidean_triangle_bin_encoder(2, 200,20,D=10000)
var_hdv = encode(encoder, var_)

var_matrix = 2*Float64.(reduce(hcat,[var_hdv[i].vector for i in traininds])) .-1
#y = 2*Float64.(reduce(hcat,[distributions_hdv[i].vector for i in traininds])) .-1
distr_matrix = -Float64.(reduce(hcat,[distributions_hdv[i].vector for i in traininds])) 
@time M = var_matrix*distr_matrix';
size(M)

p=plot()
for si in s
    prediction = RealHDV(-M'*var_hdv[si].vector)
    prediction_axis = dot(prediction, axis_hdv) #/ dot(prediction, one_hdv)
    prediction_axis = cancelnoise(prediction_axis)
    prediction_axis = prediction_axis/sum(prediction_axis)*sum(A[si,:])
    plot!(prediction_axis)
    plot!(A[si,:])
end
p

using LinearAlgebra

λ=1
@time M2 = inv(var_matrix*var_matrix'+ λ*I)*var_matrix*distr_matrix'
p=plot()
for si in s
    prediction = RealHDV(-M2'*var_hdv[si].vector)
    prediction_axis = dot(prediction, axis_hdv) #/ dot(prediction, one_hdv)
    prediction_axis = cancelnoise(prediction_axis)
    prediction_axis = prediction_axis/sum(prediction_axis)*sum(A[si,:])
    plot!(prediction_axis)
    plot!(A[si,:])
end
p