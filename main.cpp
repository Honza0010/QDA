#include <iostream>
#include <armadillo>
#include <vector>


#include "QDA.h"

using namespace std;
using namespace arma;


int main()
{

	try
	{
		QDA q(3, 4, "iris.txt");

		cout << "predicting (5.84333, 3.05733, 3.75800,	1.1993) - class: " << q.predict_class(std::vector<double>({ 5.84333333333333, 3.05733333333333, 3.75800000000000,	1.19933333333333 })) << endl;


		//cout << "predicted (1,1,1,1) " << q.predict_class(std::vector<double>({ 1,1,1,1 })) << endl; 

		cout << "predicting (1, 9, 2, 1) - class: " << q.predict_class(std::vector<double>({ 1, 9, 2, 1 })) << endl;

		cout << "predicting (5, 4, 2, 0.5) - class: " << q.predict_class(std::vector<double>({ 5, 4, 2, 0.5 })) << endl;
	}
	catch (exception e)
	{
		cout << e.what() << endl;
	}

	

	


	return 0;
}
