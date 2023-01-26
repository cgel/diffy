#include <string.h>
#include <iostream>
#include <assert.h>
#include <random>
#include <array>
#include <cmath>
using namespace std;

template <class F>
class Mat {
	public:
		int h;
		int w;
		int size;
		F * data;

		Mat(int height, int width) {
			h = height;
			w = width;
			size = h*w;
			data = new F[size];
			for (int i=0; i<size; i++ ) {*(data+i) = 0;}
		}

		Mat():Mat(1,1) {}

		Mat(Mat& A) {
			h=A.h;
			w=A.w;
			size=A.size;
			data = new F[size];
			memcpy(data, A.data, size*sizeof(F));
		}

		Mat(Mat&& A) {
			data = A.data;
			A.data = NULL;
		}

		Mat& operator=(Mat&& A) {
			h=A.h;
			w=A.w;
			size=A.size;
			data = A.data;
			A.data = NULL;
			return *this;
		}

		~Mat(){
			delete[] data;
		}

		F& ind(int i, int j){
            if (i<0 or j<0 or i>=h or j>= w) {
                throw std::invalid_argument("i, j should be in the range of h, w\n");
            }
			return *(data + i*w + j);
		}

        Mat& random(float mean, float std) {
            std::random_device rd; 
            std::mt19937 gen(rd());  
            std::normal_distribution<F> dist(mean, std); 
            for (int i=0; i<size; i++) {
                *(data+i) =  dist(gen);
            }
            return *this;
        }
        Mat& random() {
            return random(0,1);
        }

		float norm(){
			float x=0;
			for (int i=0; i<size; i++) {
				x+= pow(data[i], 2);
			}
			return sqrt(x);
		}

		void print() {
			for (int i=0; i<size; i++ ){
				cout << *(data+i);
				if ((i+1)%w==0) {cout << '\n';}
				else {cout << ", ";}
			}
		}

		Mat operator+(Mat& A) {
			assert (h == A.h && w == A.w);
			Mat M(h,w);
			for (int i=0; i<size; i++ ){
				*(M.data + i) = *(data+i) +  *(A.data+i);
			}
			return M;
		}

		Mat operator-(const Mat& A) {
			assert (h == A.h && w == A.w);
			Mat M(h,w);
			for (int i=0; i<size; i++ ){
				*(M.data + i) = *(data+i) -  *(A.data+i);
			}
			return M;
		}


		Mat operator*(Mat& A) {
			assert (w == A.h);
			Mat M(h,A.w);
			for (int i=0; i<M.h; i++ ){
				for (int j=0; j<M.w; j++ ){
					for (int k=0; k<w; k++ ){
						M.ind(i,j) += ind(i, k) *  A.ind(k,j);
					}
				}
			}
			return M;
		}

		Mat operator*(float s) {
			Mat M(h,w);
			for (int i=0; i<h; i++ ){
				for (int j=0; j<w; j++ ){
					M.ind(i,j) += s*ind(i, j);
				}
			}
			return M;
		}


		bool operator==(Mat& A) {
            if (h != A.h or w != A.w) {return false;}
			for (int i=0; i<size; i++ ){
				if (*(A.data + i) != *(data+i)) {return false;}
			}
			return true;
		}

        Mat& reshape(int new_h, int new_w) {
            if (new_h*new_w != size) { throw invalid_argument("size must match");}
            h = new_h;
            w = new_w;
            return *this;
        }

        Mat& reshape(Mat& A) {
            if (A.h*A.w != size) { throw invalid_argument("size must match");}
            h = A.h;
            w = A.w;
            return *this;
        }
};

template<class F>
Mat<F> operator*(float s, Mat<F>& M) {
    return  M*s;
}

template<class F>
Mat<F> operator-(float s, Mat<F>& M) {
    assert(M.size=1);
    Mat<F> N(1,1);
    N.ind(0,0) = M.ind(0,0) - s;
    return  N;
}
template<class F>
Mat<F> operator-(Mat<F>& M, float s) {
    return s-M;
}

template<class F>
Mat<F> operator+(float s, Mat<F>& M) {
    assert(M.size=1);
    Mat<F> N(1,1);
    N.ind(0,0) = M.ind(0,0) + s;
    return  N;
}
template<class F>
Mat<F> operator+(Mat<F>& M, float s) {
    return s+M;
}

template<class F>
Mat<F> operator==(Mat<F>& M, F s) {
	for (int i=0; i<M.size; i++){
		if(M.data[i] != s) {return false;}
	}
	return true;
}

inline const char * const BoolToString(bool b)
{
  return b ? "true" : "false";
}
void tests() {
	cout << "Test correctness of matmul \n";
	Mat<float> m(2,3);
	m.ind(0,0) = 1; m.ind(0,1) = 0; m.ind(0,2) = 1;
	m.ind(1,0) = 0; m.ind(1,1) = 1; m.ind(1,2) = 1;

	Mat<float> n(3,3);
	n.ind(0,0) = 0; n.ind(0,1) = 0; n.ind(0,2) = 0;
	n.ind(1,0) = 1; n.ind(1,1) = 1; n.ind(1,2) = 1;
	n.ind(2,0) = 0; n.ind(2,1) = 1; n.ind(2,2) = 1;

    Mat<float> x = m*n;

    Mat<float> y(2,3);
	y.ind(0,0) = 0; y.ind(0,1) = 1; y.ind(0,2) = 1;
	y.ind(1,0) = 1; y.ind(1,1) = 2; y.ind(1,2) = 2;
    cout << "x==y " << BoolToString(y == x) << '\n';
}