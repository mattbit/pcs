from cffi import FFI
ffibuilder = FFI()

ffibuilder.cdef("double runSimulation(double delta, int num_steps, double q0);")

ffibuilder.set_source("_montecarlo_sim",
r"""
    static double runSimulation(double delta, int num_steps, double q0)
    {
        int n;
        double alpha, q, q2acc = 0;

        for (n = 0; n < num_steps; n++) {
            // generate a guess
            q = q0 + ((double)random()/RAND_MAX - 0.5) * delta;

            // acceptance probability
            alpha = exp(-0.5*(pow(q, 2)-pow(q0, 2)));

            // if move is accepted
            if (alpha >= (double)random()/RAND_MAX) {
                q0 = q;
            }

            q2acc += pow(q, 2);
        }

        return q2acc/num_steps;
    }
""")

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
