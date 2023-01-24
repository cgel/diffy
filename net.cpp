#include <iostream>
#include <assert.h>
#include <tuple>
#include "./mat.cpp"
using namespace std;

// A differentiable funciton. Both, it's inputs and outputs are matrices.
class DiffFunc {
	public:
	virtual Mat<float> at(Mat<float>&) {
        throw std::logic_error("Abstract class. Shoud be calling overloaded chilid members.");
		return Mat<float>(1,1);
	}
	virtual Mat<float> differential_at(Mat<float>&) {
        throw std::logic_error("Abstract class. Shoud be calling overloaded chilid members.");
		return Mat<float>(1,1);
	}
	virtual Mat<float> at(Mat<float>&, Mat<float>&) {
        throw std::logic_error("Abstract class. Shoud be calling overloaded chilid members.");
		return Mat<float>(1,1);
	}
	virtual tuple<Mat<float>,Mat<float>> differential_at(Mat<float>&, Mat<float>&) {
        throw std::logic_error("Abstract class. Shoud be calling overloaded chilid members.");
		return {Mat<float>(1,1), Mat<float>(1,1)};
	}
};

class LinearFunc : public DiffFunc{
	public:
	Mat<float> at(Mat<float>& M, Mat<float>&x) {
		return  M*x;
	}
	tuple<Mat<float>,Mat<float>> differential_at(Mat<float>& M,Mat<float>& x) {
		// Returns two matrices, linearly approximating the change in the output given the inputs
		// Since the first input is a matrix, the linear approximation assumes it's been flattened. 
		cout << "inside LinearFunc.differential_at\n";
		Mat<float> L (M.h, M.size); // super inneficient to implement this way. This matrix will mostly be 0s
		for (int i=0; i<x.h; i++) {
			for (int j=0; j<M.h; j++) {
				L.ind(j, j*x.h + i) = x.ind(i, 0);
			}
		}
		return {L, M};
	}
};

class Max0: public DiffFunc{
	public:
	Mat<float> at(Mat<float>& x) {
		assert (x.w == 1);
		Mat<float> y(x.h, 1);
		for (int i=0; i< x.h; i++){
			y.ind(i,0) = max(x.ind(i,0), (float) 0);
		}
		return y;
	}

	Mat<float> differential_at(Mat<float>& x) {
		assert (x.w == 1);
		Mat<float> d(x.h, x.h);
		for (int i=0; i< x.h; i++){
			if (x.ind(i,0) > 0.) {
				d.ind(i,i) = 1;
			}
		}
		return d;
	}
} max0;

class SquareElementWise: public DiffFunc{
	public:
	Mat<float> at(Mat<float>& x) {
		assert (x.w == 1);
		Mat<float> y(x.h, 1);
		for (int i=0; i<x.h; i++){
			float a = x.ind(i,0);
			y.ind(i, 0) = a*a;
		}
		return y;
	}
	Mat<float> differential_at(Mat<float>& x) {
		assert (x.w == 1);
		Mat<float> d(x.h, x.h);
		for (int i=0; i< x.h; i++){
			d.ind(i,i) = x.ind(i,0);
		}
		return d;
	}
};

class Composition: public DiffFunc {
	DiffFunc** func_list;
	int length;
	public:
	Composition(DiffFunc** l, int n) {
		assert (n>=1);
		func_list = l;
		length = n;
	}

	Mat<float> at(Mat<float>& x) {
		DiffFunc* fp = *func_list;
		cout << "starting func composition with first func address " << fp << endl;
		cout << "step: 0 - \n";
		Mat<float> y=fp->at(x);
		y.print();
		for (int t=1; t< length; t++){
			cout << "step: " << t << " - ";
			fp = *(func_list+t);
			y = fp->at(y);
			y.print();
		}
		return y;
	}

	Mat<float> differential_at(Mat<float>& x) {
		DiffFunc* fp = *func_list;
		Mat<float> y=fp->at(x);
		Mat<float> d=fp->differential_at(x);
		for (int t=1; t< length; t++){
			fp = *(func_list+t);
			d = fp->differential_at(y) * d;
			y = fp->at(y);
		}
		return d;
	}
};

// class NN: public DiffFunc{
// 	Composition * cp;
	// public:
	// NN(int in_dim, int hidden_dim, int out_dim) {
	// 	int layer_num = 3+2; // 3 linear layers + 2 non linearities
	// 	DiffFunc** func_list = new DiffFunc*[layer_num]; 

	// 	LinearFunc* L1p = new LinearFunc(in_dim, hidden_dim);
	// 	*(func_list) = L1p;
	// 	*(func_list+1) = &max0;
	// 	LinearFunc* L2p = new LinearFunc(hidden_dim, hidden_dim);
	// 	*(func_list+2) = L2p;
	// 	*(func_list+3) = &max0;
	// 	LinearFunc* L3p = new LinearFunc(hidden_dim, out_dim);
	// 	*(func_list+4) = L3p;

	// 	// Initialize linear layers
	// 	L1p->random(0, 1);
	// 	L2p->random(0, 1);
	// 	L3p->random(0, 1);
	// 	cp = new Composition(func_list, layer_num);
	// }
	// Mat<float> at(Mat<float>& x) {
	// 	return cp->at(x);
	// }
	// Mat<float> differential_at(Mat<float>& x) {
	// 	return cp->differential_at(x);
	// }
// };



int main () {
	cout << "Hello! \n";
	Mat<float> m(2,3);
	m.ind(0,0) = 1; m.ind(0,1) = 0; m.ind(0,2) = 1;
	m.ind(1,0) = 0; m.ind(1,1) = 1; m.ind(1,2) = 1;

	Mat<float> n(3,3);
	n.ind(0,0) = 0; n.ind(0,1) = 0; n.ind(0,2) = 0;
	n.ind(1,0) = 1; n.ind(1,1) = 1; n.ind(1,2) = 1;
	n.ind(2,0) = 0; n.ind(2,1) = 1; n.ind(2,2) = 1;

	cout << "The matrix m \n";
	m.print();
	cout << "The matrix n\n";
	n.print();
	cout << "The product m*n x\n";
	Mat<float> r = m*n;
	r.print();

	Mat<float> x(3,1);
	x.ind(0,0) = 1; x.ind(1,0) = -1; x.ind(2,0) = 2;

	cout << "the vec x: \n";
	x.print();

	SquareElementWise square;
	Max0 max0;
	LinearFunc lin;
	DiffFunc* func_list[] = {&max0, &square};
	Composition stack(func_list, 2);
	// NN nn(3, 100, 1);

	cout << "the square(x): \n";
	square.at(x).print();
	cout << "the diff of square(x): \n";
	square.differential_at(x).print();
	cout << " ----- \n";

	cout << "the max0(x): \n";
	max0.at(x).print();
	cout << "the diff of max0(x): \n";
	max0.differential_at(x).print();
	cout << " ----- \n";

	cout << "Lin func with matrix: \n";
	m.print();
	cout << "Lin(x): \n";
	lin.at(m, x).print();
	cout << "the diff of lin(m, x): \n";
	auto [Lm, Lx] = lin.differential_at(m, x);
	cout << "linear approx as a funciton of m \n";
	Lm.print();
	cout << "linear approx as a funciton of x \n";
	Lx.print();
	cout << " ----- \n";

	// cout << "the stack(x): \n";
	// stack.at(x).print();
	// cout << "the diff of stack(x): \n";
	// stack.differential_at(x).print();
	// cout << " ----- \n";

	// cout << "the nn(x): \n";
	// nn.at(x).print();
	// cout << "the diff of stack(x): \n";
	// nn.differential_at(x).print();
	// cout << " ----- \n";


	return 0;
};