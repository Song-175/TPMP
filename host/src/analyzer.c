#include <stdlib.h>

void *an_calloc(int *dst, int count, int size) 
{
    *dst += (count * size);
    return calloc(count, size);
}
