%include <std_string.i>
%include <std_array.i>
%include <std_vector.i>
%include <std_map.i>
%include <std_shared_ptr.i>

%template(array_int_3) std::array<int,3>;
%template(array_int_6) std::array<int,6>;
%template(array_double_2) std::array<double,2>;
%template(array_double_3) std::array<double,3>;
%template(array_double_6) std::array<double,6>;
%template(vector_string) std::vector<std::string>;
%template(vector_char) std::vector<char>;
%template(vector_int) std::vector<int>;
%template(vector_long_long) std::vector<long>;
%template(vector_long) std::vector<long long>;
%template(vector_float) std::vector<float>;
%template(vector_double) std::vector<double>;
%template(vector_std_vector_int) std::vector<std::vector<int>>;
%template(map_string_bool) std::map<std::string, bool>;
%template(map_int_vector_string) std::map<int, std::vector<std::string>>;
%template(metadata_ptr) std::shared_ptr<sensei::MeshMetadata>;
