#include <string.h>
#include <iostream>
#include <assert.h>
#include <random>
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

		bool operator==(Mat& A) {
            if (h != A.h or w != A.w) {return false;}
			for (int i=0; i<size; i++ ){
				if (*(A.data + i) != *(data+i)) {return false;}
			}
			return true;
		}
};