target = main
test = test
deps = dist_opt.h
objs = dist_opt_test.cc dist_opt.cc
exps = test.cc
libs = -std=c++11 -lpthread

$(target): $(objs)
	g++ -o $(target) $(objs) $(libs)

$(test): $(exps)
	g++ -o $(test) $(exps) $(libs)

clean:
	rm test *.o
