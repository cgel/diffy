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
		Mat<float> L (M.h, M.size); // super inneficient to implement this way. This matrix will mostly be 0s
		for (int i=0; i<x.h; i++) {
			for (int j=0; j<M.h; j++) {
				L.ind(j, j*x.h + i) = x.ind(i, 0);
			}
		}
		return {L, M};
	}
} linear;

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
} square;

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

// For now we won't use polymorphism with the differentialbe NN function
class NN_3layer {
	public:
	Mat<float> at(Mat<float>& x, Mat<float>&A, Mat<float>&B, Mat<float>&C) {
		Mat<float> h = linear.at(A,x);
		h = max0.at(h);
		h = linear.at(B,h);
		h = max0.at(h);
		h = linear.at(C,h);
		return h;
	}
	tuple<Mat<float>,Mat<float>,Mat<float>,Mat<float>> differential_at(Mat<float>& x, Mat<float>&A, Mat<float>&B, Mat<float>&C) {
		bool debug_mode = false;

		Mat<float> h_by_A, h_by_B, h_by_C, h_by_x, h, h_by_h;
		tie(h_by_A, h_by_x) = linear.differential_at(A,x);
		h = linear.at(A,x);
		if (debug_mode) {cout << "lay1 preact norm: " << h.norm() << endl;}

		h_by_h = max0.differential_at(h);
		h_by_A = h_by_h * h_by_A;
		h_by_x = h_by_h * h_by_x;
		h = max0.at(h);
		if (debug_mode) {cout << "lay1 norm: " << h.norm() << endl;}

		tie(h_by_B, h_by_h) = linear.differential_at(B,h);
		h_by_A = h_by_h * h_by_A;
		h_by_x = h_by_h * h_by_x;
		h = linear.at(B,h);
		if (debug_mode) {cout << "lay2 preact norm: " << h.norm() << endl;}

		h_by_h = max0.differential_at(h);
		h_by_B = h_by_h * h_by_B;
		h_by_A = h_by_h * h_by_A;
		h_by_x = h_by_h * h_by_x;
		h = max0.at(h);
		if (debug_mode) {cout << "lay2 norm: " << h.norm() << endl;}

		tie(h_by_C, h_by_h) = linear.differential_at(C,h);
		h_by_B = h_by_h * h_by_B;
		h_by_A = h_by_h * h_by_A;
		h_by_x = h_by_h * h_by_x;
		h = linear.at(C,h);
		if (debug_mode) {cout << "output norm: " << h.norm() << endl;}

		return {h_by_x, h_by_A, h_by_B, h_by_C};
	}

	tuple<Mat<float>,Mat<float>,Mat<float>,Mat<float>> differential_at_backprop(Mat<float>& x, Mat<float>&A, Mat<float>&B, Mat<float>&C) {
		Mat<float> h1 = linear.at(A,x);
		Mat<float> h2 = max0.at(h1);
		Mat<float> h3 = linear.at(B,h2);
		Mat<float> h4 = max0.at(h3);
		Mat<float> h5 = linear.at(C,h4);

		Mat<float> h_by_A, h_by_B, h_by_C, h_by_x, h, h_by_h;
		auto [h5_by_C, h5_by_h4] = linear.differential_at(C,h4);
		auto h4_by_h3 = max0.differential_at(h3);
		Mat<float> h5_by_h3 = h5_by_h4 * h4_by_h3;
		auto [h3_by_B, h3_by_h2] = linear.differential_at(B,h2);
		Mat<float> h5_by_h2 = h5_by_h3 * h3_by_h2;
		Mat<float> h5_by_B = h5_by_h3 * h3_by_B;
		auto h2_by_h1 = max0.differential_at(h1);
		Mat<float> h5_by_h1 = h5_by_h2 * h2_by_h1;
		auto [h1_by_A, h1_by_x] = linear.differential_at(A,x);
		Mat<float> h5_by_x = h5_by_h1 * h1_by_x;
		Mat<float> h5_by_A = h5_by_h1 * h1_by_A;
		return {h5_by_x, h5_by_A, h5_by_B, h5_by_C};
	}

} nn;