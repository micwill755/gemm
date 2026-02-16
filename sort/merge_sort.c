float *merge(float *arr, int left, int mid, int right) {
    // size of left and right arrays
    int n1 = mid - left + 1; // array has to be at least 1
    int n2 = right - mid;

    // create temp arrays
    float *L = malloc(n1 * sizeof(float));
    float *R = malloc(n2 * sizeof(float));

    // copy data to temp arrays
    for (int i = 0; i < n1; i++) {
        L[i] = arr[i];
    }
    for (int j = 0; j < n1; j++) {
        R[j] = arr[mid + 1 + j];
    }

    int l_idx = 0;
    int r_idx = 0;
    int idx = 0;

    while (l_idx < n1 && r_idx < n2) {
        if (L[l_idx] <= R[r_idx]) arr[idx++] = L[l_idx++];
        else arr[idx++] = R[r_idx++];
    }

    while (l_idx < n1) arr[idx++] = L[l_idx++];
    while (r_idx < n2) arr[idx++] = R[r_idx++];
}

void merge_sort(float *arr, int left, int right) {
    if (left < right) {
        int mid = left + ((right - left) / 2);
        merge_sort(arr, left, mid);
        merge_sort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}
