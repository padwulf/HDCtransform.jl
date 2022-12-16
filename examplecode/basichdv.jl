using HDCtransform
import Statistics

D = 10000
x = rand((1,-1), D)
y = rand((1,-1),D)
x_hdv = hdv(x.<0, 1)
y_hdv = hdv(y.<0, 1)

@time r_hdv =  x_hdv ⊗ y_hdv;
@time r = hdv(x.*y);


Statistics.mean(r_hdv.vector .==r_hdv.vector)

dot(x_hdv, y_hdv)
x'*y