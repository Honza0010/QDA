#ifndef __QDA_
#define __QDA_

#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <algorithm>

#include <armadillo>

class QDA
{
	int size;
	int k;	//Number of classes
	int m;	//Dimension
	std::vector<arma::mat> data;
	std::vector<arma::mat> means;
	std::vector<arma::mat> cov_matrices;
	std::vector<double> probabs;

public:
	QDA(int k, int m, const std::string& filename);

	int predict_class(std::vector<double> input_data);

private:
	void load_file(const std::string& filename);

	void calculate_means();

	void calculate_cov_matrices();

	void calculate_probabs();

	void write()
	{
		std::cout << data[0] << std::endl;
		/*std::cout << this->size << std::endl;

		for (int i = 0; i < k; i++)
		{
			for (int j = 0; j < m; j++)
			{
				std::cout << means[i](0,j) << " ";
			}
			std::cout << std::endl;
		}*/

		for (int i = 0; i < k; i++)
		{
			std::cout << cov_matrices[i] << std::endl;
		}
	}
};


QDA::QDA(int k, int m, const std::string& filename)
	: k(k), m(m)
{
	data = std::vector<arma::mat>(k, arma::mat(0, m));
	means = std::vector<arma::mat>(k, arma::mat(1, m));
	cov_matrices = std::vector<arma::mat>(k, arma::mat());
	probabs = std::vector<double>(k, 0);

	load_file(filename);

	calculate_probabs();

	calculate_means();

	calculate_cov_matrices();

	

	write();
}



void QDA::load_file(const std::string& filename)	// Reads the file which has to be in specific format.
{													// In rows there are single vector variables and the number of class they are in
	std::ifstream file;
	file.open(filename);
	if (file.is_open())
	{
		int size_ = 0;
		int x;
		while (!file.eof())
		{
			std::vector<double> help(m + 1, 0);
			for (int i = 0; i <= this->m; i++)
			{
				if (file.eof())
				{
					throw std::logic_error("Bad format of dataset");
				}
				file >> help[i];
				if (file.fail())
				{
					throw std::logic_error("Bad values in dataset");
				}
			}

			int index = help[m] - 1;

			arma::mat& class_matrix = data[index];
			class_matrix.resize(class_matrix.n_rows + 1, m);
			for (int i = 0; i < m; i++)
			{
				class_matrix(class_matrix.n_rows - 1, i) = help[i];
			}
			size_++;
			//data[index].resize(data[index].n_rows + 1, m);
		}
		this->size = size_;
		file.close();
	}
	else
	{
		throw std::invalid_argument("File could not have been opened");
	}
}

void QDA::calculate_means()		//Calculate vector of means for each class
{
	for (int i = 0; i < this->k; i++)
	{
		means[i] = arma::mean(data[i]);
	}
}

void QDA::calculate_cov_matrices()		//Calculates cov. matrix for all classes using func. arma::cov()
{
	for (int i = 0; i < this->k; i++)
	{
		cov_matrices[i] = arma::cov(data[i]);
	}
}

void QDA::calculate_probabs()		//Pi_k = # in class k / # of all data
{
	for (int i = 0; i < k; i++)
	{
		probabs[i] = (double)data[i].n_rows / (double)this->size;
		std::cout << probabs[i] << std::endl;
	}
}


int QDA::predict_class(std::vector<double> input_data)
{
	if (input_data.size() != this->m)
	{
		throw std::logic_error("Bad dimension of input data");
	}

	arma::mat x(1, m);	//Input data for testing
	for (int i = 0; i < this->m; i++)
	{
		x(0, i) = input_data[i];
	}

	std::vector<double> deltas = std::vector<double>(this->k, 0);	//Delta for each class
																	//delta(k) = -1/2log(abs(det(cov_k)) - 1/2 (x-mean_k)^T * inv(cov_k) * (x-mean_k) + log(pi_k)
	for (int i = 0; i < k; i++)
	{
		deltas[i] = 0.0 - 1.0 / 2.0 * std::log(std::abs(arma::det(cov_matrices[i])));
		arma::mat pom = (1.0 / 2.0) * (x - means[i]) * arma::inv(cov_matrices[i]) * arma::trans(x - means[i]);
		deltas[i] -= pom(0, 0);
		deltas[i] += std::log(probabs[i]);
	}

	double max = deltas[0];
	int index = 0;				//index = argmax(deltas)
	for (int i = 0; i < k; i++)
	{
		if (max < deltas[i])
		{
			index = i;
			max = deltas[i];
		}
	}

	return (index+1);
}



#endif // !__QDA_