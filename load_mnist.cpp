#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <array>
using namespace std;

using image_type = array<int, 28*28>;

struct datapoint_type{
    image_type image;
    int label;
    void print() {
        cout << "Label: " << label << endl;
        cout << "Image: ";
        for (int k=0; k<28*28; k++) {cout << image[k] << ',';}
        cout << endl;
    }
};

// Couldn't find the original mnist files. Lecun's website requries username + password now
// The files in datasets use a different formatting

class MNIST {
    datapoint_type parse_line(string line) {
        stringstream ss(line);
        string num;
        datapoint_type datapoint;
        getline(ss, num, ',');
        // the first element contains the label
        datapoint.label = stoi(num);
        for (int i=0; i<28*28; i++) {
            getline(ss, num, ',');
            datapoint.image[i] = stoi(num);
        }
        return datapoint;
    }
    public:
    // Cannot allocate the entire array on the stack
    datapoint_type* train_datapoints;
    datapoint_type* test_datapoints;

    MNIST() {
        train_datapoints = new datapoint_type[60000];
        test_datapoints = new datapoint_type[10000];
        string line;
        ifstream trfile;
        trfile.open("./datasets/mnist_train.csv");
        if (trfile.is_open()) {
            for (int i=0; i<60000; i++) {
                getline(trfile, line);
                train_datapoints[i] = parse_line(line); 
            }
        }
        else {
            cout << "Could not read mnist_train.csv\n";
        }
        ifstream tefile;
        tefile.open("./datasets/mnist_test.csv");
        if (tefile.is_open()) {
            for (int i=0; i<10000; i++) {
                getline(tefile, line);
                test_datapoints[i] = parse_line(line); 
            }
        }
        else {
            cout << "Could not read mnist_test.csv\n";
        }
 
    }
    ~MNIST() {
        delete train_datapoints;
        delete test_datapoints;
    }
};