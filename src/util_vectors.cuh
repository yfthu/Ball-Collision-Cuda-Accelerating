/** @file  util_vectors.h
*
*   @brief contains 3-vector classes
*
*
*  @author Created by Sriram Vaidhyanathan on Thu Feb 26 2004.
*  @author Modified by Roberto Lublinerman on Mon Feb 19 2007.
*  @author Modified by Fan Yang on Dec 26 2020.
*
**/



#ifndef _RT_VECTORS_H_
#define _RT_VECTORS_H_

#include <iostream>
using namespace std;

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>


/* Class for 3-vectors */
class Vec3f
{
private:
	/** @brief Data members */
	float data[3];

public:
	/** @brief Null Constructor for class Vec2f
	*/
	__host__ __device__ Vec3f() { data[0] = data[1] = data[2] = 0; }

	/** @brief Copy Constructor for class Vec2f
	*
	*  @param V Vector to copy from
	*/
	__host__ __device__ Vec3f(const Vec3f& V) {
		data[0] = V.data[0];
		data[1] = V.data[1];
		data[2] = V.data[2];
	}

	/** @brief Constructor for class Vec2f from three floats representing the coordinates
	*
	*  @param d0 First vector coordinate
	*  @param d1 Second vector coordinate
	*  @param d2 Thrid vector coordinate
	*/
	__host__ __device__ Vec3f(float d0, float d1, float d2) {
		data[0] = d0;
		data[1] = d1;
		data[2] = d2;
	}

	/** @brief Constructor for class Vec2f as a difference from two vectors (points)
	*
	*  You can use this constructor to build a vector from two points.
	*
	*  @param V1 First vector
	*  @param V2 Second vector
	*/
	__host__ __device__ Vec3f(const Vec3f& V1, const Vec3f& V2) {
		data[0] = V1.data[0] - V2.data[0];
		data[1] = V1.data[1] - V2.data[1];
		data[2] = V1.data[2] - V2.data[2];
	}

	/** @brief Destructor
	*
	*/
	__host__ __device__ ~Vec3f() { }

	/** @brief Accessor to the three data members simultaneously
	*
	*  Load the two vector coordinates to variables d0, d1 and d2.
	*
	*  @param d0 First vector coordinate
	*  @param d1 Second vector coordinate
	*  @param d2 Second vector coordinate
	*/
	__host__ __device__ void Get(float& d0, float& d1, float& d2) const {
		d0 = data[0];
		d1 = data[1];
		d2 = data[2];
	}

	/** @brief Array like accessor to vector fields (overloading the array operator [])
	*
	*  Access the fields of a vector as if it where an array, e.g.
	*		first coordinate: vector[0]
	*
	*  @param i component to be accessed
	*/
	__host__ __device__ float operator[](int i) const {
		assert(i >= 0 && i < 3);
		return data[i];
	}

	/** @brief Standard accessor for the first coordinate
	*/
	__host__ __device__ float x() const { return data[0]; }
	/** @brief Standard accessor for the second coordinate
	*/
	__host__ __device__ float y() const { return data[1]; }
	/** @brief Standard accessor for the third coordinate
	*/
	__host__ __device__ float z() const { return data[2]; }

	/** @brief Alternative accessor for the first coordinate (red)
	*
	*	Useful if you use Vec3f to represent colors.
	*/
	__host__ __device__ float r() const { return data[0]; }
	/** @brief Alternative accessor for the second coordinate (green)
	*
	*	Useful if you use Vec3f to represent colors.
	*/
	__host__ __device__ float g() const { return data[1]; }
	/** @brief Alternative accessor for the third coordinate (blue)
	*
	*	Useful if you use Vec3f to represent colors.
	*/
	__host__ __device__ float b() const { return data[2]; }

	/** @brief Compute the length (norm) of the vector
	*/
	__host__ __device__ float Length() const {
		float l = (float)sqrt(data[0] * data[0] +
			data[1] * data[1] +
			data[2] * data[2]);
		return l;
	}
	__host__ __device__ double rayWeightLength()
	{
		return (data[0] * data[0] + data[1] * data[1] + data[2] * data[2]) / 3.0;
	}
	/** @brief Accessor to set the three fields simultaneously
	*
	*  Load the vector coordinates from variables d0,d1 and d2.
	*
	*  @param d0 First vector coordinate
	*  @param d1 Second vector coordinate
	*  @param d2 Third vector coordinate
	*/
	__host__ __device__ void Set(float d0, float d1, float d2) {
		data[0] = d0;
		data[1] = d1;
		data[2] = d2;
	}

	/** @brief Non uniform scaling of the vector
	*
	*  @param d0 scaling of the first vector coordinate
	*  @param d1 scaling of the second vector coordinate
	*  @param d2 scaling of the third vector coordinate
	*/
	__host__ __device__ void Scale(float d0, float d1, float d2) {
		data[0] *= d0;
		data[1] *= d1;
		data[2] *= d2;
	}


	/** @brief Non uniform inverse scaling of the vector
	*
	*
	*  @param d0 inverse scaling of the first vector coordinate
	*  @param d1 inverse scaling of the second vector coordinate
	*  @param d2 scaling of the third vector coordinate
	*/
	__host__ __device__ void Divide(float d0, float d1, float d2) {
		data[0] /= d0;
		data[1] /= d1;
		data[2] /= d2;
	}

	/** @brief Normalize a vector to unit length
	*/
	__host__ __device__ void Normalize() {
		float l = Length();
		if (l > 0) {
			data[0] /= l;
			data[1] /= l;
			data[2] /= l;
		}
	}

	/** @brief Negate the vector (flip the signs of each component)
	*/
	__host__ __device__ void Negate() {
		data[0] = -data[0];
		data[1] = -data[1];
		data[2] = -data[2];
	}

	/** @brief Copy operator for class Vec2f (overloading =)
	*
	*  @param V Vector to copy from
	*/
	__host__ __device__ Vec3f& operator=(const Vec3f& V) {
		data[0] = V.data[0];
		data[1] = V.data[1];
		data[2] = V.data[2];
		return *this;
	}

	/** @brief Equality comparison operator for class Vec2f (overloading ==)
	*
	*  @param V Vector to compare to
	*/
	__host__ __device__ int operator==(const Vec3f& V) {
		return ((data[0] == V.data[0]) &&
			(data[1] == V.data[1]) &&
			(data[2] == V.data[2]));
	}

	/** @brief Inequality comparison operator for class Vec2f (overloading !=)
	*
	*  @param V Vector to compare to
	*/
	__host__ __device__ int operator!=(const Vec3f& V) {
		return ((data[0] != V.data[0]) ||
			(data[1] != V.data[1]) ||
			(data[2] != V.data[2]));
	}

	/** @brief Vector addition  (overloading +=)
	*
	*  @param V Vector (point) to add
	*/
	__host__ __device__ Vec3f& operator+=(const Vec3f& V) {
		data[0] += V.data[0];
		data[1] += V.data[1];
		data[2] += V.data[2];
		return *this;
	}

	/** @brief Vector subtraction  (overloading -=)
	*
	*  @param V Vector (point) to add
	*/
	__host__ __device__ Vec3f& operator-=(const Vec3f& V) {
		data[0] -= V.data[0];
		data[1] -= V.data[1];
		data[2] -= V.data[2];
		return *this;
	}

	/** @brief Vector integer scalar multiplication  (overloading *=)
	*
	*  @param i scalar to multiply vector by
	*/
	__host__ __device__ Vec3f& operator*=(int i) {
		data[0] = float(data[0] * i);
		data[1] = float(data[1] * i);
		data[2] = float(data[2] * i);
		return *this;
	}

	/** @brief Vector float scalar multiplication  (overloading *=)
	*
	*  @param f scalar to multiply vector by
	*/
	__host__ __device__ Vec3f& operator*=(float f) {
		data[0] *= f;
		data[1] *= f;
		data[2] *= f;
		return *this;
	}

	/** @brief Vector integer scalar division  (overloading /=)
	*
	*  @param i integer scalar to divide vector by
	*/
	__host__ __device__ Vec3f& operator/=(int i) {
		data[0] = float(data[0] / i);
		data[1] = float(data[1] / i);
		data[2] = float(data[2] / i);
		return *this;
	}

	/** @brief Vector float scalar division  (overloading /=)
	*
	*  @param f float scalar to divide vector by
	*/
	__host__ __device__ Vec3f& operator/=(float f) {
		data[0] /= f;
		data[1] /= f;
		data[2] /= f;
		return *this;
	}

	/** @brief Addition of two vectors (overloading +)
	*
	*	@param v1 First addition operand
	*	@param v2 Second addition operand
	*/
	__host__ __device__ friend Vec3f operator+(const Vec3f& v1, const Vec3f& v2) {
		Vec3f v3; Add(v3, v1, v2); return v3;
	}

	/** @brief Subtraction of two vectors (overloading -)
	*
	*	@param v1 First addition operand
	*	@param v2 Second addition operand
	*/
	__host__ __device__ friend Vec3f operator-(const Vec3f& v1, const Vec3f& v2) {
		Vec3f v3; Sub(v3, v1, v2); return v3;
	}

	/** @brief Scalar vector multiplication
	*
	*	@param v1 Vector operand
	*	@param f Scalar operand
	*/
	__host__ __device__ friend Vec3f operator*(const Vec3f& v1, float f) {
		Vec3f v2; CopyScale(v2, v1, f); return v2;
	}

	__host__ __device__ Vec3f Mul(const Vec3f& v2)
	{
		Vec3f& v1 = *this;
		Vec3f v3(v1[0] * v2[0], v1[1] * v2[1], v1[2] * v2[2]);
		return v3;
	}
	/** @brief Dot (scalar) product
	*
	*/
	__host__ __device__ float Dot3(const Vec3f& V) const {
		return data[0] * V.data[0] +
			data[1] * V.data[1] +
			data[2] * V.data[2];
	}

	/** @brief Addition of two vectors
	*
	*	@param a Vector to store the results of the addition of b+c
	*	@param b First addition operand
	*	@param c Second addition operand
	*/
	__host__ __device__ friend void Add(Vec3f& a, const Vec3f& b, const Vec3f& c) {
		a.data[0] = b.data[0] + c.data[0];
		a.data[1] = b.data[1] + c.data[1];
		a.data[2] = b.data[2] + c.data[2];
	}

	/** @brief Subtraction of two vectors
	*
	*	@param a Vector to store the results of the subtraction of b0-c
	*	@param b First subtraction operand
	*	@param c Second subtraction operand
	*/
	__host__ __device__ friend void Sub(Vec3f& a, const Vec3f& b, const Vec3f& c) {
		a.data[0] = b.data[0] - c.data[0];
		a.data[1] = b.data[1] - c.data[1];
		a.data[2] = b.data[2] - c.data[2];
	}

	/** @brief Copy a scaled version of b to a (a =c*b)
	*
	*	@param a Vector to store the results of the subtraction of c*b
	*	@param b Vector operand
	*	@param c Scalar operand
	*/
	__host__ __device__ friend void CopyScale(Vec3f& a, const Vec3f& b, float c) {
		a.data[0] = b.data[0] * c;
		a.data[1] = b.data[1] * c;
		a.data[2] = b.data[2] * c;
	}

	/** @brief Add a scaled version of c to b (a = b+c*d)
	*
	*	@param a Vector to store the results of the subtraction of a= b+c*d
	*	@param b First vector operand
	*	@param c Second vector operand
	*	@param d Scalar operand
	*/
	__host__ __device__ friend void AddScale(Vec3f& a, const Vec3f& b, const Vec3f& c, float d) {
		a.data[0] = b.data[0] + c.data[0] * d;
		a.data[1] = b.data[1] + c.data[1] * d;
		a.data[2] = b.data[2] + c.data[2] * d;
	}

	/** @brief Average two vectors
	*
	*	@param a Vector to store the results of (b+c/2)
	*	@param b First vector operand
	*	@param c Second vector operand
	*/
	__host__ __device__ friend void Average(Vec3f& a, const Vec3f& b, const Vec3f& c) {
		a.data[0] = (b.data[0] + c.data[0]) * 0.5f;
		a.data[1] = (b.data[1] + c.data[1]) * 0.5f;
		a.data[2] = (b.data[2] + c.data[2]) * 0.5f;
	}

	/** @brief Compute the weighted sum of  two vectors
	*
	*	@param a Vector to store the results of (b*c+d*e)
	*	@param b First vector operand
	*	@param c First scalar operand
	*	@param d Second vector operan
	*	@param e Second scalar operand
	*/
	__host__ __device__ friend void WeightedSum(Vec3f& a, const Vec3f& b, float c, const Vec3f& d, float e) {
		a.data[0] = b.data[0] * c + d.data[0] * e;
		a.data[1] = b.data[1] * c + d.data[1] * e;
		a.data[2] = b.data[2] * c + d.data[2] * e;
	}

	/** @brief Compute the cross product of  two vectors
	*
	*	@param c Vector to store the results of (v1 x v2)
	*	@param v1 First vector operand
	*	@param v2 Second vector operan
	*/
	__host__ __device__ static void Cross3(Vec3f& c, const Vec3f& v1, const Vec3f& v2) {
		float x = v1.data[1] * v2.data[2] - v1.data[2] * v2.data[1];
		float y = v1.data[2] * v2.data[0] - v1.data[0] * v2.data[2];
		float z = v1.data[0] * v2.data[1] - v1.data[1] * v2.data[0];
		c.data[0] = x; c.data[1] = y; c.data[2] = z;
	}

	/** @brief Write the vector to a file
	*
	*	@param F pointer to a FILE structure (standard output if ommited)
	*/
	__host__ __device__ void Write(FILE* F = stdout) {
		fprintf(F, "%f %f %f\n", data[0], data[1], data[2]);
	}

};


/** @brief Ouput the vector to a stream.
*
*	Useful for debugging purposes
*/
inline ostream& operator<<(ostream& os, const Vec3f& v) {
	os << "Vec3f <" << v.x() << ", " << v.y() << ", " << v.z() << ">";
	return os;
}

/** @brief Colors can be represented by Vec3f: Color3
*
*/
typedef Vec3f Color3;

#endif
