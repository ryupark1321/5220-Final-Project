#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>


int find_arg_idx(int argc, char **argv, fs::path &p, fs::path &o)
{
  if (argc < 2 || argc % 2 == 1 || argc > 4 || strcmp(argv[1], "-h") == 0)
  {
    return -1;
  }
  int returnVal = 1;
  p = fs::path(argv[1]);
  for (int i = 2; i < argc; ++i)
  {
    if (strcmp(argv[i], "-o") == 0)
    {
      if (i != 2)
        return -1;
      else
        returnVal++;
    }
    else
    { // should be a path
      o = fs::path(argv[i]);
      returnVal++;
    }
  }
  return returnVal;
}



int main(int argc, char** argv) {
  fs::path base_dir;
  fs::path output_dir;
  // Parse input
  if (find_arg_idx(argc, argv, base_dir, output_dir) < 0)
  {
    std::cout << "Usage: <program> <images/weights directory> -o[optional] <path to output directory>" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "-h: see this help" << std::endl;
    std::cout << "-o <path>: path to output directory" << std::endl;
    return 0;
  }
}