
### Standard Bubble Sort Algorithm

```cpp
void bubbleSort(std::vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                std::swap(arr[j], arr[j + 1]);
            }
        }
    }
}
```

### Optimized Bubble Sort Algorithm

```cpp
void optimizedBubbleSort(std::vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        bool swapped = false;
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                std::swap(arr[j], arr[j + 1]);
                swapped = true;
            }
        }
        if (!swapped) {
            break;
        }
    }
}
```

### Selection Sort

```cpp
void selectionSort(std::vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        int minIndex = i;
        for (int j = i + 1; j < n; j++) {
            if (arr[j] < arr[minIndex]) {
                minIndex = j;
            }
        }
        std::swap(arr[i], arr[minIndex]);
    }
}
```

### Insertion Sort Iterative Algorithm

```cpp
void insertionSort(std::vector<int>& arr) {
    int n = arr.size();
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}
```

### Recursive Insertion Sort

```cpp
void recursiveInsertionSort(std::vector<int>& arr, int n) {
    if (n <= 1) return;
    recursiveInsertionSort(arr, n - 1);
    int key = arr[n - 1];
    int j = n - 2;
    while (j >= 0 && arr[j] > key) {
        arr[j + 1] = arr[j];
        j--;
    }
    arr[j + 1] = key;
}
```

### MergeSort Recursive

```cpp
void merge(std::vector<int>& arr, int left, int mid, int right) {
    int n1 = mid - left + 1, n2 = right - mid;
    std::vector<int> L(n1), R(n2);
  
    for (int i = 0; i < n1; i++) L[i] = arr[left + i];
    for (int i = 0; i < n2; i++) R[i] = arr[mid + 1 + i];
  
    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
    }
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
}
void mergeSort(std::vector<int>& arr, int left, int right) {
    if (left >= right) return;
    int mid = left + (right - left) / 2;
    mergeSort(arr, left, mid);
    mergeSort(arr, mid + 1, right);
    merge(arr, left, mid, right);
}
```

### MergeSort Iterative

```cpp
void iterativeMergeSort(std::vector<int>& arr) {
    int n = arr.size();
    for (int curr_size = 1; curr_size < n; curr_size *= 2) {
        for (int left_start = 0; left_start < n - 1; left_start += 2 * curr_size) {
            int mid = std::min(left_start + curr_size - 1, n - 1);
            int right_end = std::min(left_start + 2 * curr_size - 1, n - 1);
            merge(arr, left_start, mid, right_end);
        }
    }
}
```
### Quick Sort (Lomuto Partition)
```cpp
int lomutoPartition(std::vector<int>& arr, int low, int high) {
    int pivot = arr[high], i = low;
    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) std::swap(arr[i++], arr[j]);
    }
    std::swap(arr[i], arr[high]);
    return i;
}
void quickSortLomuto(std::vector<int>& arr, int low, int high) {
    if (low < high) {
        int pi = lomutoPartition(arr, low, high);
        quickSortLomuto(arr, low, pi - 1);
        quickSortLomuto(arr, pi + 1, high);
    }
}
void iterative_quick_sort(vector<int>& arr) {
    int n = arr.size();
    stack<pair<int, int>> stk;

    stk.push({0, n -1});

    while(!(stk.empty())) {
        int low = stk.top().first;
        int high = stk.top().second;
        stk.pop();

        if (low < high) {
            int pi = partition(arr, low, high);
                stk.push({low, pi - 1});
                stk.push({pi + 1, high});
        }
    }
}
```

### Quick Sort (Hoare Partition)

```cpp
int partition(vector<int>& arr, int low, int high)  {
    int pivot = arr[low];
    int i = low - 1, j = high + 1;

    while (true)    {
        do {i++;} while (arr[i] < pivot);
        do {j--;} while (arr[j] > pivot);
        if (i >= j) return j;
        swap(arr[i], arr[j]);
    }
}

void recursive_quicksort(vector<int>& arr, int low, int high)   {
    if (low < high) {
        int pi = partition(arr, low, high);
        recursive_quicksort(arr, low, pi);
        recursive_quicksort(arr, pi + 1, high);
    }
}
```

### Quick Sort Randomised

```cpp
int partition(vector<int>& arr, int low, int high)  {
    int pivot = arr[high];
    int i = low - 1;

    for(int j = low; j < high; j++) {
        if(arr[j] < pivot)  {
            i++;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);
    return i + 1;
}

int randomised_partition(vector<int>& arr, int low, int high)   {
    int random = low + rand() % (high - low + 1);
    swap(arr[random], arr[high]);
    return partition(arr, low, high);
}

void quicksort(vector<int>& arr, int low, int high)    {
    if (low < high)
    {
        int pi = randomised_partition(arr, low, high);
        quicksort(arr, low, pi - 1);
        quicksort(arr, pi + 1, high);
    }
    
}
```
### Basic Heap Sort (Max Heap)
```cpp
void heapify(int arr[], int n, int i) {
    int largest = i;
    int left = 2 * i + 1; 
    int right = 2 * i + 2; 

    if (left < n && arr[left] > arr[largest])
        largest = left;

    if (right < n && arr[right] > arr[largest])
        largest = right;

    if (largest != i) {
        swap(arr[i], arr[largest]);
        heapify(arr, n, largest);
    }
}

void heapSort(int arr[], int n) {
    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(arr, n, i);

    for (int i = n - 1; i > 0; i--) {
        swap(arr[0], arr[i]); 
        heapify(arr, i, 0);  
    }
}
```

### Heap Sort Using Priority Queue (STL)

```cpp
void heapSort(vector<int>& arr) {
    priority_queue<int> maxHeap;

    for (int num : arr)
        maxHeap.push(num);

    for (int i = arr.size() - 1; i >= 0; i--) {
        arr[i] = maxHeap.top();
        maxHeap.pop();
    }
}
```

###  Heap Sort Using Min Heap
```cpp
void heapSort(vector<int>& arr) {
    priority_queue<int, vector<int>, greater<int>> minHeap;

    for (int num : arr)
        minHeap.push(num);

    for (int i = 0; i < arr.size(); i++) {
        arr[i] = minHeap.top();
        minHeap.pop();
    }
}
```

### Iterative Heap Sort (Without Recursion)

```cpp
void heapify(int arr[], int n, int i) {
    while (true) {
        int largest = i;
        int left = 2 * i + 1;
        int right = 2 * i + 2;

        if (left < n && arr[left] > arr[largest])
            largest = left;

        if (right < n && arr[right] > arr[largest])
            largest = right;

        if (largest == i)
            break;

        swap(arr[i], arr[largest]);
        i = largest;
    }
}

void heapSort(int arr[], int n) {
    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(arr, n, i);

    for (int i = n - 1; i > 0; i--) {
        swap(arr[0], arr[i]);
        heapify(arr, i, 0);
    }
}
```

### Counting Sort
```cpp
void countingsort(vector<int>& arr) {
    if (arr.empty()) return;

    int minval = arr[0], maxval = arr[0];
    for(int num: arr)   {
        if (num < minval) minval = num;
        if (num > maxval) maxval = num;
    }

    int range = maxval - minval + 1;
    vector<int> count(range, 0);
    vector<int> output(arr.size());

    for(int num : arr)  count[num - minval]++;

    for(int i = 1; i < range; i++)  count[i] += count[i - 1];

    for (int i = arr.size() - 1; i >= 0; i--)   {
        output[count[arr[i] - minval] - 1] = arr[i];
        count[arr[i] - minval]--;
    }

    arr = output;
}
```

### Radix Sort LSD
```cpp
void countingsort(vector<int>& arr, int exp)    {
    int n = arr.size();
    vector<int> count(10, 0);
    vector<int> output(n);

    for (int num: arr)  count[(num / exp) % 10]++;

    for (int i = 1; i < 10; i++)    count[i] += count[i - 1];

    for (int i = n - 1; i >= 0; i--)    {
        int digit = (arr[i] / exp) % 10;
        output[count[digit] - 1] = arr[i];
        count[digit]--;
    }
    arr = output;
}

void radixsort(vector<int>& arr)    {
    int maxval = arr[0];
    for (int num: arr)  {
        if (maxval < num)   {
            maxval = num;
        }
    }

    for (int exp = 1; maxval / exp > 0; exp *= 10) {
        countingsort(arr, exp);
    }
}
```

### Radix Sort MSD
```cpp
int getDigit(int num, int place) {
    return (num / place) % 10;
}

int getMax(const vector<int>& arr) {
    int maxVal = arr[0];
    for (int num : arr)
        if (num > maxVal)
            maxVal = num;
    return maxVal;
}

void msdRadixSort(vector<int>& arr, int left, int right, int place) {
    if (left >= right || place == 0)
        return; // Base case

    vector<vector<int>> buckets(10); 

    for (int i = left; i <= right; i++)
        buckets[getDigit(arr[i], place)].push_back(arr[i]);

    int index = left;
    
    for (int i = 0; i < 10; i++) {
        if (!buckets[i].empty()) {
            for (int num : buckets[i])
                arr[index++] = num;
            msdRadixSort(arr, index - buckets[i].size(), index - 1, place / 10);
        }
    }
}

void radixSortMSD(vector<int>& arr) {
    int maxNum = getMax(arr);
    int maxPlace = pow(10, to_string(maxNum).length() - 1); 
    msdRadixSort(arr, 0, arr.size() - 1, maxPlace);
}
```

### Bucket Sort (Uniform Distribution)
```cpp
void bucketSort(vector<float>& arr) {
    int n = arr.size();
    if (n == 0) return;

    // Step 1: Create n empty buckets
    vector<vector<float>> buckets(n);

    // Step 2: Place elements in respective buckets
    for (float num : arr) {
        int index = n * num; // Map value to bucket index
        buckets[index].push_back(num);
    }

    // Step 3: Sort individual buckets
    for (auto& bucket : buckets)
        sort(bucket.begin(), bucket.end()); // Using std::sort (can use Insertion Sort for small data)

    // Step 4: Concatenate sorted buckets back into original array
    int idx = 0;
    for (auto& bucket : buckets)
        for (float num : bucket)
            arr[idx++] = num;
}
```

### Bucket Sort Non-uniform distribution

```cpp
int partition(vector<float>& arr, int low, int high) {
    float pivot = arr[high];
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (arr[j] <= pivot) {
            swap(arr[++i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);
    return i + 1;
}

// QuickSort function
void quickSort(vector<float>& arr, int low, int high) {
    if (low < high) {
        int pivot = partition(arr, low, high);
        quickSort(arr, low, pivot - 1);
        quickSort(arr, pivot + 1, high);
    }
}

// Function to perform optimized Bucket Sort
void optimizedBucketSort(vector<float>& arr) {
    int n = arr.size();
    if (n == 0) return;

    // Step 1: Find min and max values to determine bucket range
    float minVal = *min_element(arr.begin(), arr.end());
    float maxVal = *max_element(arr.begin(), arr.end());
    
    float range = maxVal - minVal;
    int bucketCount = n;  // Use n buckets for optimal performance

    // Step 2: Create empty buckets
    vector<vector<float>> buckets(bucketCount);

    // Step 3: Distribute elements into buckets
    for (float num : arr) {
        int index = (int)((num - minVal) / range * (bucketCount - 1)); // Adaptive mapping
        buckets[index].push_back(num);
    }

    // Step 4: Sort individual buckets
    int idx = 0;
    for (auto& bucket : buckets) {
        if (bucket.size() > 1) {
            quickSort(bucket, 0, bucket.size() - 1); // Use QuickSort for large buckets
        }
        for (float num : bucket)
            arr[idx++] = num;
    }
}
```

### Maximum Sum Subarray Kadane’s Algorithm (DP)
```cpp
int maxSubArray(vector<int>& nums) {
    int maxSum = INT_MIN;  
    int currentSum = 0;     

    for (int num : nums) {
        currentSum += num;  
        maxSum = max(maxSum, currentSum); 

        if (currentSum < num)  
            currentSum = num;
    }
    
    return maxSum;
}
```

### Maximum Sum Subarray Divide and Conquer
```cpp
int crossing_sum(vector<int>& arr, int left, int mid, int right) {
    int left_sum = INT_MIN, sum = 0;

    for (int i = mid; i >= left; i--) {
        sum += arr[i];
        left_sum = max(left_sum, sum);
    }

    int right_sum = INT_MIN;
    sum = 0;

    for (int i = mid + 1; i <= right; i++) {
        sum += arr[i];
        right_sum = max(right_sum, sum);
    }

    return left_sum + right_sum;
}

int maximum_subarray_sum(vector<int>& arr, int left, int right) {
    if (left > right) return INT_MIN; // Edge case: invalid range
    if (left == right) return arr[left];

    int mid = left + (right - left) / 2;

    int left_max = maximum_subarray_sum(arr, left, mid);
    int right_max = maximum_subarray_sum(arr, mid + 1, right);
    int cross_max = crossing_sum(arr, left, mid, right);

    return max({left_max, right_max, cross_max});
}
```

### Finding Maximum and Minimum Divide and Conquer 
```cpp
struct MinMax   {
    int min;
    int max;
};

MinMax findMinMax(int arr[], int low, int high) {
    MinMax result, left, right;

    if (low == high)    {
        result.min = arr[low];
        result.max = arr[high];
        return result;
    }

    if (high == low + 1)    {
        if (arr[low] < arr[high])   {
            result.min = arr[low];
            result.max = arr[high];
        }   else    {
            result.max = arr[low];
            result.min = arr[high];
        }
        return result;
    }

    int mid = low + (high - low) / 2;
    left = findMinMax(arr, low, mid);
    right = findMinMax(arr, mid + 1, high);

    result.min = min(left.min, right.min);
    result.max = max(left.max, right.max);

    return result;
}
```

### Binary Search Iterative
```cpp
int binarysearch(int arr[], int n, int target)  {
    int left = 0, right = n - 1;
    while (left <= right)   {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target)    {
            return mid;
        }   else if (arr[mid] < target) {
            left = mid + 1;
        }  else     {
            right = mid - 1;
        }
    }
    return - 1;
}
```

### Binary Search Recursive
```cpp
int binarysearch(vector<int>& arr, int target, int low, int high)   {
    if (low > high) return -1;
    int mid = low + (high - low) / 2;
    if (arr[mid] == target) return mid;
    if (arr[mid] < target) return binarysearch(arr, target, mid + 1, high);
    return binarysearch(arr, target, low, mid - 1);
}
```

### Binary Search Order-Agnostic
```cpp
int agnosticbinarysearch(vector<int>& arr, int target)  {
    bool isascending = (arr.front() < arr.back()) ? true : false;

    int left = 0, right = arr.size() - 1;
    
    while (left <= right)   {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target) return mid;

        if (isascending)    {
            if (arr[mid] < target)  {
                left = mid + 1;
            }   else    {
                right = mid - 1;
            }
        }   else    {
            if (arr[mid] > target)  {
                left = mid + 1;
            }   else    {
                right = mid - 1;
            }
        }
    }
   return - 1;
}
```

### Minimum Distance Between Two Points	Brute Force
```cpp
struct Point    {
    int x; int y;
};

double distance(Point p1, Point p2) {
    return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}

double bruteforce(Point point[], int n) {
    double mind = DBL_MAX;
    for(int i = 0; i < n; i++)  {
        for (int j = i + 1; j < n; j++) {
            mind = min(mind, distance(point[i], point[j]));
        }
    }
    return mind;
}
```

### Minimum Distance Between Two Points Divide and Conquer

```cpp
struct Point {
    int x, y;
};

bool compareX(const Point& p1, const Point& p2) { return p1.x < p2.x; }
bool compareY(const Point& p1, const Point& p2) { return p1.y < p2.y; }

double distance(const Point& p1, const Point& p2) {
    return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}

double stripClosest(vector<Point>& strip, double d) {
    double minDist = d;
    sort(strip.begin(), strip.end(), compareY); // Sort by Y

    for (int i = 0; i < strip.size(); ++i) {
        for (int j = i + 1; j < strip.size() && (strip[j].y - strip[i].y) < minDist; ++j) {
            minDist = min(minDist, distance(strip[i], strip[j]));
        }
    }
    return minDist;
}

double closestPairUtil(Point points[], int left, int right) {
    if (right - left <= 3) {
        double minDist = DBL_MAX;
        for (int i = left; i <= right; i++) {
            for (int j = i + 1; j <= right; j++) {
                minDist = min(minDist, distance(points[i], points[j]));
            }
        }
        return minDist;
    }

    int mid = (left + right) / 2;
    Point midPoint = points[mid];


    double dLeft = closestPairUtil(points, left, mid);
    double dRight = closestPairUtil(points, mid + 1, right);
    double d = min(dLeft, dRight);


    vector<Point> strip;
    for (int i = left; i <= right; i++) {
        if (abs(points[i].x - midPoint.x) < d)
            strip.push_back(points[i]);
    }

    return min(d, stripClosest(strip, d));
}

double closestPair(Point points[], int n) {
    sort(points, points + n, compareX);
    return closestPairUtil(points, 0, n - 1);
}
```

### Long Integer Multiplication	Naïve Approach
```cpp
string multiply(string num1, string num2) {
    int len1 = num1.size();
    int len2 = num2.size();
    vector<int> result(len1 + len2, 0); 
 
    reverse(num1.begin(), num1.end());
    reverse(num2.begin(), num2.end());

    for (int i = 0; i < len1; i++) {
        for (int j = 0; j < len2; j++) {
            result[i + j] += (num1[i] - '0') * (num2[j] - '0');
            result[i + j + 1] += result[i + j] / 10; 
            result[i + j] %= 10;
        }
    }

    string product = "";
    for (int i = result.size() - 1; i >= 0; i--) {
        if (!(product.empty() && result[i] == 0)) { 
            product += (result[i] + '0');
        }
    }

    return product.empty() ? "0" : product;
}
```

### Long Integer Multiplication	Karatsuba Algorithm

```cpp
string addStrings(string num1, string num2) {
    int carry = 0, sum;
    string result = "";

    int i = num1.size() - 1, j = num2.size() - 1;
    while (i >= 0 || j >= 0 || carry) {
        sum = carry;
        if (i >= 0) sum += num1[i--] - '0';
        if (j >= 0) sum += num2[j--] - '0';
        result += (sum % 10) + '0';
        carry = sum / 10;
    }

    reverse(result.begin(), result.end());
    return result;
}

// Function to multiply two large numbers using Karatsuba Algorithm
string karatsuba(string X, string Y) {
    int n = max(X.size(), Y.size());

    // Base case: If numbers are small, multiply directly
    if (n == 1) return to_string((X[0] - '0') * (Y[0] - '0'));

    // Make numbers equal in length by padding with zeros
    while (X.size() < n) X = "0" + X;
    while (Y.size() < n) Y = "0" + Y;

    // Split numbers into two halves
    int mid = n / 2;
    string A = X.substr(0, mid);
    string B = X.substr(mid);
    string C = Y.substr(0, mid);
    string D = Y.substr(mid);

    // Recursively compute three multiplications
    string AC = karatsuba(A, C);
    string BD = karatsuba(B, D);
    string sumA_B = addStrings(A, B);
    string sumC_D = addStrings(C, D);
    string AD_BC = karatsuba(sumA_B, sumC_D);
    
    // AD + BC = (A+B)(C+D) - AC - BD
    AD_BC = addStrings(AD_BC, "-" + AC);
    AD_BC = addStrings(AD_BC, "-" + BD);

    // Compute the final result
    // Multiply AC by 10^(2m)
    for (int i = 0; i < 2 * (n - mid); i++) AC += "0"; 
    for (int i = 0; i < (n - mid); i++) AD_BC += "0"; 

    return addStrings(addStrings(AC, AD_BC), BD);
}
```
