// helper function to write the rendered image as PPM file
static void writePPM(const char *fileName, int size_x, int size_y, const uint32_t *pixel) {
  using namespace std;

  FILE *file = fopen(fileName, "wb");
  if (!file) {
    fprintf(stderr, "fopen('%s', 'wb') failed: %d", fileName, errno);
    return;
  }
  fprintf(file, "P6\n%i %i\n255\n", size_x, size_y);
  unsigned char *out = (unsigned char *)alloca(3 * size_x);
  for (int y = 0; y < size_y; y++) {
    const unsigned char *in =
        (const unsigned char *)&pixel[(size_y - 1 - y) * size_x];
    for (int x = 0; x < size_x; x++) {
      out[3 * x + 0] = in[4 * x + 0];
      out[3 * x + 1] = in[4 * x + 1];
      out[3 * x + 2] = in[4 * x + 2];
    }
    fwrite(out, 3 * size_x, sizeof(char), file);
  }
  fprintf(file, "\n");
  fclose(file);
}
