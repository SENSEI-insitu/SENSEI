#!/usr/bin/env perl

# This script tries to find deprecated attribute data classes and
# methods and warns the user whenever it finds them. It also suggests
# possible modification to bring code up to date.

use Getopt::Long;

if (!GetOptions("language:s" => \$language,
		"verbose" => \$verbose,
		"help" => \$help,
		"print-messages" => \$print))
{
    die;
}

if (!$language)
{
    $language = "c++";
}

if ( !$print && ($#ARGV < 0 || $help) )
{
    print "Usage: $0 [--language {c++ | python | java}] ",
          "[--verbose] [--help] [--print-messages] file1 [file2 ...]\n";
    exit;
}


%cxxmessageids = (
		  'svtkScalars\*' => 0,
		  'svtkVectors\*' => 1,
		  'svtkNormals\*' => 2,
		  'svtkTCoords\*' => 3,
		  'svtkTensors\*' => 4,
		  'svtkScalars::New[ \t]*\(' => 5,
		  'svtkVectors::New[ \t]*\(' => 6,
		  'svtkNormals::New[ \t]*\(' => 7,
		  'svtkTCoords::New[ \t]*\(' => 8,
		  'svtkTensors::New[ \t]*\(' => 9,
		  '->GetScalar[ \t]*\(' => 10,
		  '->GetVector[ \t]*\(' => 11,
		  '->GetNormal[ \t]*\(' => 12,
		  '->GetTCoord[ \t]*\(' => 13,
		  '->GetTensor[ \t]*\(' => 14,
		  '->SetScalar[ \t]*\(' => 15,
		  '->SetVector[ \t]*\(' => 16,
		  '->SetNormal[ \t]*\(' => 17,
		  '->SetTCoord[ \t]*\(' => 18,
		  '->SetTensor[ \t]*\(' => 19,
		  '->GetScalars[ \t]*\([a-zA-Z]+.*\)' => 20,
		  '->GetVectors[ \t]*\([a-zA-Z]+.*\)' => 21,
		  '->GetNormals[ \t]*\([a-zA-Z]+.*\)' => 22,
		  '->GetTCoords[ \t]*\([a-zA-Z]+.*\)' => 23,
		  '->GetTensors[ \t]*\([a-zA-Z]+.*\)' => 24,
		  '->InsertScalar[ \t]*\(' => 25,
		  '->InsertVector[ \t]*\(' => 26,
		  '->InsertNormal[ \t]*\(' => 27,
		  '->InsertTCoord[ \t]*\(' => 28,
		  '->InsertTensor[ \t]*\(' => 29,
		  '->InsertNextScalar[ \t]*\(' => 30,
		  '->InsertNextVector[ \t]*\(' => 31,
		  '->InsertNextNormal[ \t]*\(' => 32,
		  '->InsertNextTCoord[ \t]*\(' => 33,
		  '->InsertNextTensor[ \t]*\(' => 34,
		  '->GetActiveScalars' => 35,
		  '->GetActiveVectors' => 36,
		  '->GetActiveNormals' => 37,
		  '->GetActiveTCoords' => 38,
		  '->GetActiveTensors' => 39,
		  '->GetNumberOfScalars' => 40,
		  '->GetNumberOfVectors' => 41,
		  '->GetNumberOfNormals' => 42,
		  '->GetNumberOfTCoords' => 43,
		  '->GetNumberOfTensors' => 44,
		  '->SetNumberOfScalars' => 40,
		  '->SetNumberOfVectors' => 41,
		  '->SetNumberOfNormals' => 42,
		  '->SetNumberOfTCoords' => 43,
		  '->SetNumberOfTensors' => 44,
		  );

%pythonmessageids = (
		     'svtkScalars[ \t]*\(\)' => 5,
		     'svtkVectors[ \t]*\(\)' => 6,
		     'svtkNormals[ \t]*\(\)' => 7,
		     'svtkTCoords[ \t]*\(\)' => 8,
		     'svtkTensors[ \t]*\(\)' => 9,
		     '\.GetScalar[ \t]*\(' => 10,
		     '\.GetVector[ \t]*\(' => 11,
		     '\.GetNormal[ \t]*\(' => 12,
		     '\.GetTCoord[ \t]*\(' => 13,
		     '\.GetTensor[ \t]*\(' => 14,
		     '\.SetScalar[ \t]*\(' => 15,
		     '\.SetVector[ \t]*\(' => 16,
		     '\.SetNormal[ \t]*\(' => 17,
		     '\.SetTCoord[ \t]*\(' => 18,
		     '\.SetTensor[ \t]*\(' => 19,
		     '.GetScalars[ \t]*\([a-zA-Z]+.*\)' => 20,
		     '.GetVectors[ \t]*\([a-zA-Z]+.*\)' => 21,
		     '.GetNormals[ \t]*\([a-zA-Z]+.*\)' => 22,
		     '.GetTCoords[ \t]*\([a-zA-Z]+.*\)' => 23,
		     '.GetTensors[ \t]*\([a-zA-Z]+.*\)' => 24,
		     '.InsertScalar[ \t]*\(' => 25,
		     '.InsertVector[ \t]*\(' => 26,
		     '.InsertNormal[ \t]*\(' => 27,
		     '.InsertTCoord[ \t]*\(' => 28,
		     '.InsertTensor[ \t]*\(' => 29,
		     '.InsertNextScalar[ \t]*\(' => 30,
		     '.InsertNextVector[ \t]*\(' => 31,
		     '.InsertNextNormal[ \t]*\(' => 32,
		     '.InsertNextTCoord[ \t]*\(' => 33,
		     '.InsertNextTensor[ \t]*\(' => 34,
		     '.GetActiveScalars' => 35,
		     '.GetActiveVectors' => 36,
		     '.GetActiveNormals' => 37,
		     '.GetActiveTCoords' => 38,
		     '.GetActiveTensors' => 39,
		     '.GetNumberOfScalars' => 40,
		     '.GetNumberOfVectors' => 41,
		     '.GetNumberOfNormals' => 42,
		     '.GetNumberOfTCoords' => 43,
		     '.GetNumberOfTensors' => 44,
		     '.SetNumberOfScalars' => 40,
		     '.SetNumberOfVectors' => 41,
		     '.SetNumberOfNormals' => 42,
		     '.SetNumberOfTCoords' => 43,
		     '.SetNumberOfTensors' => 44,
		     );

%javamessageids = (
		   'new[ \t]+svtkScalars[ \t]*\(\)' => 5,
		   'new[ \t]+svtkVectors[ \t]*\(\)' => 6,
		   'new[ \t]+svtkNormals[ \t]*\(\)' => 7,
		   'new[ \t]+svtkTCoords[ \t]*\(\)' => 8,
		   'new[ \t]+svtkTensors[ \t]*\(\)' => 9,
		   '\.GetScalar[ \t]*\(' => 10,
		   '\.GetVector[ \t]*\(' => 11,
		   '\.GetNormal[ \t]*\(' => 12,
		   '\.GetTCoord[ \t]*\(' => 13,
		   '\.GetTensor[ \t]*\(' => 14,
		   '\.SetScalar[ \t]*\(' => 15,
		   '\.SetVector[ \t]*\(' => 16,
		   '\.SetNormal[ \t]*\(' => 17,
		   '\.SetTCoord[ \t]*\(' => 18,
		   '\.SetTensor[ \t]*\(' => 19,
		   '.GetScalars[ \t]*\([a-zA-Z]+.*\)' => 20,
		   '.GetVectors[ \t]*\([a-zA-Z]+.*\)' => 21,
		   '.GetNormals[ \t]*\([a-zA-Z]+.*\)' => 22,
		   '.GetTCoords[ \t]*\([a-zA-Z]+.*\)' => 23,
		   '.GetTensors[ \t]*\([a-zA-Z]+.*\)' => 24,
		   '.InsertScalar[ \t]*\(' => 25,
		   '.InsertVector[ \t]*\(' => 26,
		   '.InsertNormal[ \t]*\(' => 27,
		   '.InsertTCoord[ \t]*\(' => 28,
		   '.InsertTensor[ \t]*\(' => 29,
		   '.InsertNextScalar[ \t]*\(' => 30,
		   '.InsertNextVector[ \t]*\(' => 31,
		   '.InsertNextNormal[ \t]*\(' => 32,
		   '.InsertNextTCoord[ \t]*\(' => 33,
		   '.InsertNextTensor[ \t]*\(' => 34,
		   '.GetActiveScalars' => 35,
		   '.GetActiveVectors' => 36,
		   '.GetActiveNormals' => 37,
		   '.GetActiveTCoords' => 38,
		   '.GetActiveTensors' => 39,
		   '.GetNumberOfScalars' => 40,
		   '.GetNumberOfVectors' => 41,
		   '.GetNumberOfNormals' => 42,
		   '.GetNumberOfTCoords' => 43,
		   '.GetNumberOfTensors' => 44,
		   '.SetNumberOfScalars' => 40,
		   '.SetNumberOfVectors' => 41,
		   '.SetNumberOfNormals' => 42,
		   '.SetNumberOfTCoords' => 43,
		   '.SetNumberOfTensors' => 44,
		   );

if ($language eq "c++")
{
    %messageids = %cxxmessageids;
}
elsif($language eq "python")
{
    %messageids = %pythonmessageids;
}
elsif($language eq "java")
{
    %messageids = %javamessageids;
}
else
{
    die "Unsupported language: $language.\n";
}
@messages = (
	     "> Encountered svtkScalars* : svtkScalars has been\n".
	     "> deprecated. You should use svtkDataArray or one\n".
	     "> of it's subclasses.\n",
	     "> Encountered svtkVectors* : svtkVectors has been\n".
	     "> deprecated. You should use svtkDataArray or one\n".
	     "> of it's subclasses.\n",
	     "> Encountered svtkNormals* : svtkNormals has been\n".
	     "> deprecated. You should use svtkDataArray or one\n".
	     "> of it's subclasses.\n",
	     "> Encountered svtkTCoords* : svtkTCoords has been\n".
	     "> deprecated. You should use svtkDataArray or one\n".
	     "> of it's subclasses.\n",
	     "> Encountered svtkTensors* : svtkTensors has been\n".
	     "> deprecated. You should use svtkDataArray or one\n".
	     "> of it's subclasses.\n",
	     "> Encountered svtkScalars constructor: svtkScalars has been\n".
	     "> deprecated. You should use svtkDataArray or one\n".
	     "> of it's subclasses.\n",
	     "> Encountered svtkVectors constructor: svtkVectors has been\n".
	     "> deprecated. You should use svtkDataArray or one\n".
	     "> of it's subclasses. Note that you have to explicitly set the\n".
	     "> number of components. For example (in Tcl):\n".
	     "> svtkFloatArray vectors\n".
	     "> vectors SetNumberOfComponents 3\n",
	     "> Encountered svtkNormals constructor: svtkNormals has been\n".
	     "> deprecated. You should use svtkDataArray or one\n".
	     "> of it's subclasses.Note that you have to explicitly set the\n".
	     "> number of components. For example (in Tcl):\n".
	     "> svtkFloatArray normals\n".
	     "> normals SetNumberOfComponents 3\n",
	     "> Encountered svtkTCoords constructor: svtkTCoords has been\n".
	     "> deprecated. You should use svtkDataArray or one\n".
	     "> of it's subclasses.Note that you have to explicitly set the\n".
	     "> number of components. For example (in Tcl):\n".
	     "> svtkFloatArray tc\n".
	     "> tc SetNumberOfComponents 2\n",
	     "> Encountered svtkTensors constructor: svtkTensors has been\n".
	     "> deprecated. You should use svtkDataArray or one\n".
	     "> of it's subclasses.Note that you have to explicitly set the\n".
	     "> number of components. For example (in Tcl):\n".
	     "> svtkFloatArray tensors\n".
	     "> tensors SetNumberOfComponents 9\n",
	     "> Encountered svtkScalars::GetScalar() : svtkScalars has been\n".
	     "> deprecated. You should replace this object with a\n".
	     "> svtkDataArray or one of it's subclasses and use GetComponent(id, component)\n".
	     "> instead of GetScalar(id)\n" ,
	     "> Encountered svtkVectors::GetVector(): svtkVectors has been\n".
	     "> deprecated. You should replace this object with a\n".
	     "> svtkDataArray or one of it's subclasses and use GetTuple(id)\n".
	     "> instead of GetVector(id)\n" ,
	     "> Encountered svtkNormals::GetNormal(): svtkNormals has been\n".
	     "> deprecated. You should replace this object with a\n".
	     "> svtkDataArray or one of it's subclasses and use GetTuple(id)\n".
	     "> instead of GetNormal(id)\n" ,
	     "> Encountered svtkTCoords::GetTCoord(): svtkTCoords has been\n".
	     "> deprecated. You should replace this object with a\n".
	     "> svtkDataArray or one of it's subclasses and use GetTuple(id)\n".
	     "> instead of GetTCoord(id)\n" ,
	     "> Encountered svtkTensors::GetTensors(): svtkTensors has been\n".
	     "> deprecated. You should replace this object with a\n".
	     "> svtkDataArray or one of it's subclasses and use GetTuple(id)\n".
	     "> instead of GetTensor(id)\n" ,
	     "> Encountered svtkScalars::SetScalar() : svtkScalars has been\n".
	     "> deprecated. You should replace this object with a\n".
	     "> svtkDataArray or one of it's subclasses and use SetComponent(id, component, value)\n".
	     "> instead of GetScalar(id)\n" ,
	     "> Encountered svtkVectors::SetVector(): svtkVectors has been\n".
	     "> deprecated. You should replace this object with a\n".
	     "> svtkDataArray or one of it's subclasses and use SetTuple(id, v)\n".
	     "> instead of SetVector(id, v)\n" ,
	     "> Encountered svtkNormals::SetNormal(): svtkNormals has been\n".
	     "> deprecated. You should replace this object with a\n".
	     "> svtkDataArray or one of it's subclasses and use SetTuple(id,v)\n".
	     "> instead of SetNormal(id,v)\n" ,
	     "> Encountered svtkTCoords::SetTCoord(): svtkTCoords has been\n".
	     "> deprecated. You should replace this object with a\n".
	     "> svtkDataArray or one of it's subclasses and use SetTuple(id,v)\n".
	     "> instead of SetTCoord(id,v)\n" ,
	     "> Encountered svtkTensors::SetTensors(): svtkTensors has been\n".
	     "> deprecated. You should replace this object with a\n".
	     "> svtkDataArray or one of it's subclasses and use SetTuple(id,v)\n".
	     "> instead of GetTensor(id,v)\n" ,
	     "> Encountered svtkScalars::GetScalars() : svtkScalars has been\n".
	     "> deprecated. You should replace this object with a\n".
	     "> svtkDataArray or one of it's subclasses and use GetTuples()\n".
	     "> instead of GetScalars(id). Note that, unlike GetScalars(),\n".
	     "> GetTuples() requires that enough memory is allocated in the\n".
	     "> target array. See the documentation of svtkDataArray for more\n".
	     "> information.\n",
	     "> Encountered svtkVectors::GetVectors() : svtkVectors has been\n".
	     "> deprecated. You should replace this object with a\n".
	     "> svtkDataArray or one of it's subclasses and use GetTuples()\n".
	     "> instead of GetVectors(id). Note that, unlike GetVectors(),\n".
	     "> GetTuples() requires that enough memory is allocated in the\n".
	     "> target array. See the documentation of svtkDataArray for more\n".
	     "> information.\n",
	     "> Encountered svtkNormals::GetNormals() : svtkNormals has been\n".
	     "> deprecated. You should replace this object with a\n".
	     "> svtkDataArray or one of it's subclasses and use GetTuples()\n".
	     "> instead of GetNormals(id). Note that, unlike GetNormals(),\n".
	     "> GetTuples() requires that enough memory is allocated in the\n".
	     "> target array. See the documentation of svtkDataArray for more\n".
	     "> information.\n",
	     "> Encountered svtkTCoords::GetTCoords() : svtkTCoords has been\n".
	     "> deprecated. You should replace this object with a\n".
	     "> svtkDataArray or one of it's subclasses and use GetTuples()\n".
	     "> instead of GetTCoords(id). Note that, unlike GetTCoords(),\n".
	     "> GetTuples() requires that enough memory is allocated in the\n".
	     "> target array. See the documentation of svtkDataArray for more\n".
	     "> information.\n",
	     "> Encountered svtkTensors::GetTensors() : svtkTensors has been\n".
	     "> deprecated. You should replace this object with a\n".
	     "> svtkDataArray or one of it's subclasses and use GetTuples()\n".
	     "> instead of GetTensors(id). Note that, unlike GetTensors(),\n".
	     "> GetTuples() requires that enough memory is allocated in the\n".
	     "> target array. See the documentation of svtkDataArray for more\n".
	     "> information.\n",
	     "> Encountered svtkScalars::InsertScalar() :  svtkScalars has been\n".
	     "> deprecated. You should replace this object with a\n".
	     "> svtkDataArray or one of it's subclasses and use InsertComponent(),\n ".
	     "> InsertValue() or InsertTuple1() instead of InsertScalar()\n",
	     "> Encountered svtkVectors::InsertVector() :  svtkVectors has been\n".
	     "> deprecated. You should replace this object with a\n".
	     "> svtkDataArray or one of it's subclasses and use InsertTuple(),\n ".
	     "> or InsertTuple3() instead of InsertVector()\n",
	     "> Encountered svtkNormals::InsertNormal() :  svtkNormals has been\n".
	     "> deprecated. You should replace this object with a\n".
	     "> svtkDataArray or one of it's subclasses and use InsertTuple(),\n ".
	     "> or InsertTuple3() instead of InsertNormal()\n",
	     "> Encountered svtkTCoords::InsertTCoord() :  svtkTCoords has been\n".
	     "> deprecated. You should replace this object with a\n".
	     "> svtkDataArray or one of it's subclasses and use InsertTuple(),\n ".
	     "> or InsertTuple2() instead of InsertTCoord()\n",
	     "> Encountered svtkTensors::InsertTensor() :  svtkTensors has been\n".
	     "> deprecated. You should replace this object with a\n".
	     "> svtkDataArray or one of it's subclasses and use InsertTuple(),\n ".
	     "> or InsertTuple9() instead of InsertTensor()\n",
	     "> Encountered svtkScalars::InsertNextScalar() :  svtkScalars has been\n".
	     "> deprecated. You should replace this object with a\n".
	     "> svtkDataArray or one of it's subclasses and use InsertNextComponent(),\n ".
	     "> InsertNextValue() or InsertNextTuple1() instead of InsertNextScalar()\n",
	     "> Encountered svtkVectors::InsertNextVector() :  svtkVectors has been\n".
	     "> deprecated. You should replace this object with a\n".
	     "> svtkDataArray or one of it's subclasses and use InsertNextTuple(),\n ".
	     "> or InsertNextTuple3() instead of InsertNextVector()\n",
	     "> Encountered svtkNormals::InsertNextNormal() :  svtkNormals has been\n".
	     "> deprecated. You should replace this object with a\n".
	     "> svtkDataArray or one of it's subclasses and use InsertNextTuple(),\n ".
	     "> or InsertNextTuple3() instead of InsertNextNormal()\n",
	     "> Encountered svtkTCoords::InsertNextTCoord() :  svtkTCoords has been\n".
	     "> deprecated. You should replace this object with a\n".
	     "> svtkDataArray or one of it's subclasses and use InsertNextTuple(),\n ".
	     "> or InsertNextTuple2() instead of InsertNextTCoord()\n",
	     "> Encountered svtkTensors::InsertNextTensor() :  svtkTensors has been\n".
	     "> deprecated. You should replace this object with a\n".
	     "> svtkDataArray or one of it's subclasses and use InsertNextTuple(),\n ".
	     "> or InsertNextTuple9() instead of InsertNextTensor()\n",
	     "> Replace svtkDataSetAttributes::GetActiveScalars() with \n".
	     "> svtkDataSetAttributes::GetScalars()\n",
	     "> Replace svtkDataSetAttributes::GetActiveVectors() with \n".
	     "> svtkDataSetAttributes::GetVectors()\n",
	     "> Replace svtkDataSetAttributes::GetActiveNormals() with \n".
	     "> svtkDataSetAttributes::GetNormals()\n",
	     "> Replace svtkDataSetAttributes::GetActiveTCoords() with \n".
	     "> svtkDataSetAttributes::GetTCoords()\n",
	     "> Replace svtkDataSetAttributes::GetActiveTensors() with \n".
	     "> svtkDataSetAttributes::GetTensors()\n",
	     "> Encountered svtkScalars::Set/GetNumberOfScalars() : svtkScalars has been\n".
	     "> deprecated. You should replace this object with a\n".
	     "> svtkDataArray or one of it's subclasses and use Set/GetNumberOfTuples()\n".
	     "> instead of Set/GetNumberOfScalars().\n",
	     "> Encountered svtkVectors::Set/GetNumberOfVectors() : svtkVectors has been\n".
	     "> deprecated. You should replace this object with a\n".
	     "> svtkDataArray or one of it's subclasses and use Set/GetNumberOfTuples()\n".
	     "> instead of Set/GetNumberOfVectors().\n",
	     "> Encountered svtkNormals::Set/GetNumberOfNormals() : svtkNormals has been\n".
	     "> deprecated. You should replace this object with a\n".
	     "> svtkDataArray or one of it's subclasses and use Set/GetNumberOfTuples()\n".
	     "> instead of Set/GetNumberOfNormals().\n",
	     "> Encountered svtkTCoords::Set/GetNumberOfTCoords() : svtkTCoords has been\n".
	     "> deprecated. You should replace this object with a\n".
	     "> svtkDataArray or one of it's subclasses and use Set/GetNumberOfTuples()\n".
	     "> instead of Set/GetNumberOfTCoords().\n",
	     "> Encountered svtkTensors::Set/GetNumberOfTensors() : svtkTensors has been\n".
	     "> deprecated. You should replace this object with a\n".
	     "> svtkDataArray or one of it's subclasses and use Set/GetNumberOfTuples()\n".
	     "> instead of Set/GetNumberOfTensors().\n",
	     );


if ( $print )
{
    $i = 0;
    foreach $key (@messages)
    {
	print "Message id $i:\n";
	print $key, "\n";
	$i++;
    }
    exit 0;
}

foreach $filename (@ARGV)
{
    open(FPTR, "<$filename") or die "Could not open file $filename";
    if ($verbose)
    {
	print "Processing file: $filename\n";
    }
    $i = 1;
    while (<FPTR>)
    {
	$line = $_;
	foreach $key (keys %messageids)
	{
	    if ( $line =~ m($key) )
	    {
		chomp $line;
		if ($verbose)
		{
		    print ">> File $filename line $i: ",
		          "\n$messages[$messageids{$key}]\n";
		}
		else
		{
		    print ">> File $filename line $i: ",
	              	  "Message $messageids{$key}\n";
		}
		last;
	    }
	}
	$i++;
    }
}

