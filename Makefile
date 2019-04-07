target = test
deps = dist_opt.h
objs = dist_opt_test.cc dist_opt.cc
libs = -std=c++11 -lpthread

$(target): $(objs)
	g++ -o $(target) $(objs) $(libs)

clean:
	rm test $(objs)
