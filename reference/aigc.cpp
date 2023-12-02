#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

int main() {
  // Open the image file.
  ifstream image("image.png");

  // Check if the file opened successfully.
  if (!image.is_open()) {
    cout << "Error opening image file." << endl;
    return 1;
  }

  // Get the size of the image.
  int width, height;
  image.seekg(0, ios::end);
  image.tellg();
  image.seekg(0, ios::beg);
  image >> width >> height;

  // Create a vector to store the image data.
  vector<unsigned char> image_data(width * height * 4);

  // Read the image data into the vector.
  image.read((char *)image_data.data(), image.gcount());

  // Close the image file.
  image.close();

  // Print the image data.
  for (int i = 0; i < image_data.size(); i++) {
    cout << image_data[i] << " ";
  }
  cout << endl;

  return 0;
}
