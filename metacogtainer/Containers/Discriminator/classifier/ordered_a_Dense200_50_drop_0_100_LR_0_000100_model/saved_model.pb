ЮН
■╙
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
;
Elu
features"T
activations"T"
Ttype:
2
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
┴
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ии
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.9.12v2.9.0-18-gd8ce9f9c3018Ч▄
|
Adam/Output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/Output/bias/v
u
&Adam/Output/bias/v/Read/ReadVariableOpReadVariableOpAdam/Output/bias/v*
_output_shapes
:*
dtype0
Д
Adam/Output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*%
shared_nameAdam/Output/kernel/v
}
(Adam/Output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Output/kernel/v*
_output_shapes

:2*
dtype0
О
Adam/FullyConnected1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*,
shared_nameAdam/FullyConnected1/bias/v
З
/Adam/FullyConnected1/bias/v/Read/ReadVariableOpReadVariableOpAdam/FullyConnected1/bias/v*
_output_shapes
:2*
dtype0
Ч
Adam/FullyConnected1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╚2*.
shared_nameAdam/FullyConnected1/kernel/v
Р
1Adam/FullyConnected1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/FullyConnected1/kernel/v*
_output_shapes
:	╚2*
dtype0
П
Adam/FullyConnected0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*,
shared_nameAdam/FullyConnected0/bias/v
И
/Adam/FullyConnected0/bias/v/Read/ReadVariableOpReadVariableOpAdam/FullyConnected0/bias/v*
_output_shapes	
:╚*
dtype0
Ч
Adam/FullyConnected0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@╚*.
shared_nameAdam/FullyConnected0/kernel/v
Р
1Adam/FullyConnected0/kernel/v/Read/ReadVariableOpReadVariableOpAdam/FullyConnected0/kernel/v*
_output_shapes
:	@╚*
dtype0
|
Adam/Output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/Output/bias/m
u
&Adam/Output/bias/m/Read/ReadVariableOpReadVariableOpAdam/Output/bias/m*
_output_shapes
:*
dtype0
Д
Adam/Output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*%
shared_nameAdam/Output/kernel/m
}
(Adam/Output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Output/kernel/m*
_output_shapes

:2*
dtype0
О
Adam/FullyConnected1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*,
shared_nameAdam/FullyConnected1/bias/m
З
/Adam/FullyConnected1/bias/m/Read/ReadVariableOpReadVariableOpAdam/FullyConnected1/bias/m*
_output_shapes
:2*
dtype0
Ч
Adam/FullyConnected1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╚2*.
shared_nameAdam/FullyConnected1/kernel/m
Р
1Adam/FullyConnected1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/FullyConnected1/kernel/m*
_output_shapes
:	╚2*
dtype0
П
Adam/FullyConnected0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*,
shared_nameAdam/FullyConnected0/bias/m
И
/Adam/FullyConnected0/bias/m/Read/ReadVariableOpReadVariableOpAdam/FullyConnected0/bias/m*
_output_shapes	
:╚*
dtype0
Ч
Adam/FullyConnected0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@╚*.
shared_nameAdam/FullyConnected0/kernel/m
Р
1Adam/FullyConnected0/kernel/m/Read/ReadVariableOpReadVariableOpAdam/FullyConnected0/kernel/m*
_output_shapes
:	@╚*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
n
Output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameOutput/bias
g
Output/bias/Read/ReadVariableOpReadVariableOpOutput/bias*
_output_shapes
:*
dtype0
v
Output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*
shared_nameOutput/kernel
o
!Output/kernel/Read/ReadVariableOpReadVariableOpOutput/kernel*
_output_shapes

:2*
dtype0
А
FullyConnected1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*%
shared_nameFullyConnected1/bias
y
(FullyConnected1/bias/Read/ReadVariableOpReadVariableOpFullyConnected1/bias*
_output_shapes
:2*
dtype0
Й
FullyConnected1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╚2*'
shared_nameFullyConnected1/kernel
В
*FullyConnected1/kernel/Read/ReadVariableOpReadVariableOpFullyConnected1/kernel*
_output_shapes
:	╚2*
dtype0
Б
FullyConnected0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*%
shared_nameFullyConnected0/bias
z
(FullyConnected0/bias/Read/ReadVariableOpReadVariableOpFullyConnected0/bias*
_output_shapes	
:╚*
dtype0
Й
FullyConnected0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@╚*'
shared_nameFullyConnected0/kernel
В
*FullyConnected0/kernel/Read/ReadVariableOpReadVariableOpFullyConnected0/kernel*
_output_shapes
:	@╚*
dtype0

NoOpNoOp
щ8
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*д8
valueЪ8BЧ8 BР8
ш
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
ж
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
е
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator* 
ж
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

%kernel
&bias*
е
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses
-_random_generator* 
ж
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses

4kernel
5bias*
.
0
1
%2
&3
44
55*
.
0
1
%2
&3
44
55*
* 
░
6non_trainable_variables

7layers
8metrics
9layer_regularization_losses
:layer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
;trace_0
<trace_1
=trace_2
>trace_3* 
6
?trace_0
@trace_1
Atrace_2
Btrace_3* 
* 
╖
Citer

Dbeta_1

Ebeta_2
	Fdecay
Glearning_ratem{m|%m}&m~4m5mАvБvВ%vГ&vД4vЕ5vЖ*

Hserving_default* 

0
1*

0
1*
* 
У
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Ntrace_0* 

Otrace_0* 
f`
VARIABLE_VALUEFullyConnected0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEFullyConnected0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
С
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

Utrace_0
Vtrace_1* 

Wtrace_0
Xtrace_1* 
* 

%0
&1*

%0
&1*
* 
У
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*

^trace_0* 

_trace_0* 
f`
VARIABLE_VALUEFullyConnected1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEFullyConnected1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
С
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses* 

etrace_0
ftrace_1* 

gtrace_0
htrace_1* 
* 

40
51*

40
51*
* 
У
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses*

ntrace_0* 

otrace_0* 
]W
VARIABLE_VALUEOutput/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEOutput/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
.
0
1
2
3
4
5*

p0
q1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
r	variables
s	keras_api
	ttotal
	ucount*
H
v	variables
w	keras_api
	xtotal
	ycount
z
_fn_kwargs*

t0
u1*

r	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

x0
y1*

v	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
КГ
VARIABLE_VALUEAdam/FullyConnected0/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Е
VARIABLE_VALUEAdam/FullyConnected0/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
КГ
VARIABLE_VALUEAdam/FullyConnected1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Е
VARIABLE_VALUEAdam/FullyConnected1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/Output/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/Output/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
КГ
VARIABLE_VALUEAdam/FullyConnected0/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Е
VARIABLE_VALUEAdam/FullyConnected0/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
КГ
VARIABLE_VALUEAdam/FullyConnected1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Е
VARIABLE_VALUEAdam/FullyConnected1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/Output/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/Output/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
x
serving_default_inputPlaceholder*'
_output_shapes
:         @*
dtype0*
shape:         @
╕
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputFullyConnected0/kernelFullyConnected0/biasFullyConnected1/kernelFullyConnected1/biasOutput/kernelOutput/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *.
f)R'
%__inference_signature_wrapper_5316804
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ў

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*FullyConnected0/kernel/Read/ReadVariableOp(FullyConnected0/bias/Read/ReadVariableOp*FullyConnected1/kernel/Read/ReadVariableOp(FullyConnected1/bias/Read/ReadVariableOp!Output/kernel/Read/ReadVariableOpOutput/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp1Adam/FullyConnected0/kernel/m/Read/ReadVariableOp/Adam/FullyConnected0/bias/m/Read/ReadVariableOp1Adam/FullyConnected1/kernel/m/Read/ReadVariableOp/Adam/FullyConnected1/bias/m/Read/ReadVariableOp(Adam/Output/kernel/m/Read/ReadVariableOp&Adam/Output/bias/m/Read/ReadVariableOp1Adam/FullyConnected0/kernel/v/Read/ReadVariableOp/Adam/FullyConnected0/bias/v/Read/ReadVariableOp1Adam/FullyConnected1/kernel/v/Read/ReadVariableOp/Adam/FullyConnected1/bias/v/Read/ReadVariableOp(Adam/Output/kernel/v/Read/ReadVariableOp&Adam/Output/bias/v/Read/ReadVariableOpConst*(
Tin!
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *)
f$R"
 __inference__traced_save_5317124
╓
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameFullyConnected0/kernelFullyConnected0/biasFullyConnected1/kernelFullyConnected1/biasOutput/kernelOutput/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/FullyConnected0/kernel/mAdam/FullyConnected0/bias/mAdam/FullyConnected1/kernel/mAdam/FullyConnected1/bias/mAdam/Output/kernel/mAdam/Output/bias/mAdam/FullyConnected0/kernel/vAdam/FullyConnected0/bias/vAdam/FullyConnected1/kernel/vAdam/FullyConnected1/bias/vAdam/Output/kernel/vAdam/Output/bias/v*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *,
f'R%
#__inference__traced_restore_5317215д╪
э
Д
)__inference_model_2_layer_call_fn_5316737	
input
unknown:	@╚
	unknown_0:	╚
	unknown_1:	╚2
	unknown_2:2
	unknown_3:2
	unknown_4:
identityИвStatefulPartitionedCallП
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_model_2_layer_call_and_return_conditional_losses_5316705o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:         @

_user_specified_nameinput
▐
╛
D__inference_model_2_layer_call_and_return_conditional_losses_5316758	
input*
fullyconnected0_5316740:	@╚&
fullyconnected0_5316742:	╚*
fullyconnected1_5316746:	╚2%
fullyconnected1_5316748:2 
output_5316752:2
output_5316754:
identityИв'FullyConnected0/StatefulPartitionedCallв'FullyConnected1/StatefulPartitionedCallвOutput/StatefulPartitionedCallТ
'FullyConnected0/StatefulPartitionedCallStatefulPartitionedCallinputfullyconnected0_5316740fullyconnected0_5316742*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╚*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_FullyConnected0_layer_call_and_return_conditional_losses_5316519ы
dropout_00/PartitionedCallPartitionedCall0FullyConnected0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╚* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_dropout_00_layer_call_and_return_conditional_losses_5316530п
'FullyConnected1/StatefulPartitionedCallStatefulPartitionedCall#dropout_00/PartitionedCall:output:0fullyconnected1_5316746fullyconnected1_5316748*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_FullyConnected1_layer_call_and_return_conditional_losses_5316543ъ
dropout_01/PartitionedCallPartitionedCall0FullyConnected1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_dropout_01_layer_call_and_return_conditional_losses_5316554Л
Output/StatefulPartitionedCallStatefulPartitionedCall#dropout_01/PartitionedCall:output:0output_5316752output_5316754*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_Output_layer_call_and_return_conditional_losses_5316567v
IdentityIdentity'Output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ╗
NoOpNoOp(^FullyConnected0/StatefulPartitionedCall(^FullyConnected1/StatefulPartitionedCall^Output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : : : : : 2R
'FullyConnected0/StatefulPartitionedCall'FullyConnected0/StatefulPartitionedCall2R
'FullyConnected1/StatefulPartitionedCall'FullyConnected1/StatefulPartitionedCall2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall:N J
'
_output_shapes
:         @

_user_specified_nameinput
и

 
L__inference_FullyConnected0_layer_call_and_return_conditional_losses_5316926

inputs1
matmul_readvariableop_resource:	@╚.
biasadd_readvariableop_resource:	╚
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@╚*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚O
EluEluBiasAdd:output:0*
T0*(
_output_shapes
:         ╚a
IdentityIdentityElu:activations:0^NoOp*
T0*(
_output_shapes
:         ╚w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
▐
e
G__inference_dropout_00_layer_call_and_return_conditional_losses_5316530

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         ╚\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ╚"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ╚:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs
¤	
f
G__inference_dropout_00_layer_call_and_return_conditional_losses_5316652

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         ╚C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         ╚*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=з
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ╚p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ╚j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         ╚Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         ╚"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ╚:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs
д

■
L__inference_FullyConnected1_layer_call_and_return_conditional_losses_5316543

inputs1
matmul_readvariableop_resource:	╚2-
biasadd_readvariableop_resource:2
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	╚2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2N
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:         2`
IdentityIdentityElu:activations:0^NoOp*
T0*'
_output_shapes
:         2w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ╚: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs
Я

Ї
C__inference_Output_layer_call_and_return_conditional_losses_5316567

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         2
 
_user_specified_nameinputs
╟
А
%__inference_signature_wrapper_5316804	
input
unknown:	@╚
	unknown_0:	╚
	unknown_1:	╚2
	unknown_2:2
	unknown_3:2
	unknown_4:
identityИвStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *+
f&R$
"__inference__wrapped_model_5316501o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:         @

_user_specified_nameinput
э
Д
)__inference_model_2_layer_call_fn_5316589	
input
unknown:	@╚
	unknown_0:	╚
	unknown_1:	╚2
	unknown_2:2
	unknown_3:2
	unknown_4:
identityИвStatefulPartitionedCallП
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_model_2_layer_call_and_return_conditional_losses_5316574o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:         @

_user_specified_nameinput
▐
Й
D__inference_model_2_layer_call_and_return_conditional_losses_5316705

inputs*
fullyconnected0_5316687:	@╚&
fullyconnected0_5316689:	╚*
fullyconnected1_5316693:	╚2%
fullyconnected1_5316695:2 
output_5316699:2
output_5316701:
identityИв'FullyConnected0/StatefulPartitionedCallв'FullyConnected1/StatefulPartitionedCallвOutput/StatefulPartitionedCallв"dropout_00/StatefulPartitionedCallв"dropout_01/StatefulPartitionedCallУ
'FullyConnected0/StatefulPartitionedCallStatefulPartitionedCallinputsfullyconnected0_5316687fullyconnected0_5316689*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╚*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_FullyConnected0_layer_call_and_return_conditional_losses_5316519√
"dropout_00/StatefulPartitionedCallStatefulPartitionedCall0FullyConnected0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╚* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_dropout_00_layer_call_and_return_conditional_losses_5316652╖
'FullyConnected1/StatefulPartitionedCallStatefulPartitionedCall+dropout_00/StatefulPartitionedCall:output:0fullyconnected1_5316693fullyconnected1_5316695*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_FullyConnected1_layer_call_and_return_conditional_losses_5316543Я
"dropout_01/StatefulPartitionedCallStatefulPartitionedCall0FullyConnected1/StatefulPartitionedCall:output:0#^dropout_00/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_dropout_01_layer_call_and_return_conditional_losses_5316619У
Output/StatefulPartitionedCallStatefulPartitionedCall+dropout_01/StatefulPartitionedCall:output:0output_5316699output_5316701*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_Output_layer_call_and_return_conditional_losses_5316567v
IdentityIdentity'Output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Е
NoOpNoOp(^FullyConnected0/StatefulPartitionedCall(^FullyConnected1/StatefulPartitionedCall^Output/StatefulPartitionedCall#^dropout_00/StatefulPartitionedCall#^dropout_01/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : : : : : 2R
'FullyConnected0/StatefulPartitionedCall'FullyConnected0/StatefulPartitionedCall2R
'FullyConnected1/StatefulPartitionedCall'FullyConnected1/StatefulPartitionedCall2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall2H
"dropout_00/StatefulPartitionedCall"dropout_00/StatefulPartitionedCall2H
"dropout_01/StatefulPartitionedCall"dropout_01/StatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
╪
Я
1__inference_FullyConnected1_layer_call_fn_5316962

inputs
unknown:	╚2
	unknown_0:2
identityИвStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_FullyConnected1_layer_call_and_return_conditional_losses_5316543o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ╚: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs
Х 
╕
D__inference_model_2_layer_call_and_return_conditional_losses_5316865

inputsA
.fullyconnected0_matmul_readvariableop_resource:	@╚>
/fullyconnected0_biasadd_readvariableop_resource:	╚A
.fullyconnected1_matmul_readvariableop_resource:	╚2=
/fullyconnected1_biasadd_readvariableop_resource:27
%output_matmul_readvariableop_resource:24
&output_biasadd_readvariableop_resource:
identityИв&FullyConnected0/BiasAdd/ReadVariableOpв%FullyConnected0/MatMul/ReadVariableOpв&FullyConnected1/BiasAdd/ReadVariableOpв%FullyConnected1/MatMul/ReadVariableOpвOutput/BiasAdd/ReadVariableOpвOutput/MatMul/ReadVariableOpХ
%FullyConnected0/MatMul/ReadVariableOpReadVariableOp.fullyconnected0_matmul_readvariableop_resource*
_output_shapes
:	@╚*
dtype0К
FullyConnected0/MatMulMatMulinputs-FullyConnected0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚У
&FullyConnected0/BiasAdd/ReadVariableOpReadVariableOp/fullyconnected0_biasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype0з
FullyConnected0/BiasAddBiasAdd FullyConnected0/MatMul:product:0.FullyConnected0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚o
FullyConnected0/EluElu FullyConnected0/BiasAdd:output:0*
T0*(
_output_shapes
:         ╚u
dropout_00/IdentityIdentity!FullyConnected0/Elu:activations:0*
T0*(
_output_shapes
:         ╚Х
%FullyConnected1/MatMul/ReadVariableOpReadVariableOp.fullyconnected1_matmul_readvariableop_resource*
_output_shapes
:	╚2*
dtype0Я
FullyConnected1/MatMulMatMuldropout_00/Identity:output:0-FullyConnected1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2Т
&FullyConnected1/BiasAdd/ReadVariableOpReadVariableOp/fullyconnected1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0ж
FullyConnected1/BiasAddBiasAdd FullyConnected1/MatMul:product:0.FullyConnected1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2n
FullyConnected1/EluElu FullyConnected1/BiasAdd:output:0*
T0*'
_output_shapes
:         2t
dropout_01/IdentityIdentity!FullyConnected1/Elu:activations:0*
T0*'
_output_shapes
:         2В
Output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0Н
Output/MatMulMatMuldropout_01/Identity:output:0$Output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         А
Output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Л
Output/BiasAddBiasAddOutput/MatMul:product:0%Output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d
Output/SoftmaxSoftmaxOutput/BiasAdd:output:0*
T0*'
_output_shapes
:         g
IdentityIdentityOutput/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         з
NoOpNoOp'^FullyConnected0/BiasAdd/ReadVariableOp&^FullyConnected0/MatMul/ReadVariableOp'^FullyConnected1/BiasAdd/ReadVariableOp&^FullyConnected1/MatMul/ReadVariableOp^Output/BiasAdd/ReadVariableOp^Output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : : : : : 2P
&FullyConnected0/BiasAdd/ReadVariableOp&FullyConnected0/BiasAdd/ReadVariableOp2N
%FullyConnected0/MatMul/ReadVariableOp%FullyConnected0/MatMul/ReadVariableOp2P
&FullyConnected1/BiasAdd/ReadVariableOp&FullyConnected1/BiasAdd/ReadVariableOp2N
%FullyConnected1/MatMul/ReadVariableOp%FullyConnected1/MatMul/ReadVariableOp2>
Output/BiasAdd/ReadVariableOpOutput/BiasAdd/ReadVariableOp2<
Output/MatMul/ReadVariableOpOutput/MatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
в$
ї
"__inference__wrapped_model_5316501	
inputI
6model_2_fullyconnected0_matmul_readvariableop_resource:	@╚F
7model_2_fullyconnected0_biasadd_readvariableop_resource:	╚I
6model_2_fullyconnected1_matmul_readvariableop_resource:	╚2E
7model_2_fullyconnected1_biasadd_readvariableop_resource:2?
-model_2_output_matmul_readvariableop_resource:2<
.model_2_output_biasadd_readvariableop_resource:
identityИв.model_2/FullyConnected0/BiasAdd/ReadVariableOpв-model_2/FullyConnected0/MatMul/ReadVariableOpв.model_2/FullyConnected1/BiasAdd/ReadVariableOpв-model_2/FullyConnected1/MatMul/ReadVariableOpв%model_2/Output/BiasAdd/ReadVariableOpв$model_2/Output/MatMul/ReadVariableOpе
-model_2/FullyConnected0/MatMul/ReadVariableOpReadVariableOp6model_2_fullyconnected0_matmul_readvariableop_resource*
_output_shapes
:	@╚*
dtype0Щ
model_2/FullyConnected0/MatMulMatMulinput5model_2/FullyConnected0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚г
.model_2/FullyConnected0/BiasAdd/ReadVariableOpReadVariableOp7model_2_fullyconnected0_biasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype0┐
model_2/FullyConnected0/BiasAddBiasAdd(model_2/FullyConnected0/MatMul:product:06model_2/FullyConnected0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚
model_2/FullyConnected0/EluElu(model_2/FullyConnected0/BiasAdd:output:0*
T0*(
_output_shapes
:         ╚Е
model_2/dropout_00/IdentityIdentity)model_2/FullyConnected0/Elu:activations:0*
T0*(
_output_shapes
:         ╚е
-model_2/FullyConnected1/MatMul/ReadVariableOpReadVariableOp6model_2_fullyconnected1_matmul_readvariableop_resource*
_output_shapes
:	╚2*
dtype0╖
model_2/FullyConnected1/MatMulMatMul$model_2/dropout_00/Identity:output:05model_2/FullyConnected1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2в
.model_2/FullyConnected1/BiasAdd/ReadVariableOpReadVariableOp7model_2_fullyconnected1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0╛
model_2/FullyConnected1/BiasAddBiasAdd(model_2/FullyConnected1/MatMul:product:06model_2/FullyConnected1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2~
model_2/FullyConnected1/EluElu(model_2/FullyConnected1/BiasAdd:output:0*
T0*'
_output_shapes
:         2Д
model_2/dropout_01/IdentityIdentity)model_2/FullyConnected1/Elu:activations:0*
T0*'
_output_shapes
:         2Т
$model_2/Output/MatMul/ReadVariableOpReadVariableOp-model_2_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0е
model_2/Output/MatMulMatMul$model_2/dropout_01/Identity:output:0,model_2/Output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Р
%model_2/Output/BiasAdd/ReadVariableOpReadVariableOp.model_2_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0г
model_2/Output/BiasAddBiasAddmodel_2/Output/MatMul:product:0-model_2/Output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         t
model_2/Output/SoftmaxSoftmaxmodel_2/Output/BiasAdd:output:0*
T0*'
_output_shapes
:         o
IdentityIdentity model_2/Output/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         ╫
NoOpNoOp/^model_2/FullyConnected0/BiasAdd/ReadVariableOp.^model_2/FullyConnected0/MatMul/ReadVariableOp/^model_2/FullyConnected1/BiasAdd/ReadVariableOp.^model_2/FullyConnected1/MatMul/ReadVariableOp&^model_2/Output/BiasAdd/ReadVariableOp%^model_2/Output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : : : : : 2`
.model_2/FullyConnected0/BiasAdd/ReadVariableOp.model_2/FullyConnected0/BiasAdd/ReadVariableOp2^
-model_2/FullyConnected0/MatMul/ReadVariableOp-model_2/FullyConnected0/MatMul/ReadVariableOp2`
.model_2/FullyConnected1/BiasAdd/ReadVariableOp.model_2/FullyConnected1/BiasAdd/ReadVariableOp2^
-model_2/FullyConnected1/MatMul/ReadVariableOp-model_2/FullyConnected1/MatMul/ReadVariableOp2N
%model_2/Output/BiasAdd/ReadVariableOp%model_2/Output/BiasAdd/ReadVariableOp2L
$model_2/Output/MatMul/ReadVariableOp$model_2/Output/MatMul/ReadVariableOp:N J
'
_output_shapes
:         @

_user_specified_nameinput
Ё
Е
)__inference_model_2_layer_call_fn_5316838

inputs
unknown:	@╚
	unknown_0:	╚
	unknown_1:	╚2
	unknown_2:2
	unknown_3:2
	unknown_4:
identityИвStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_model_2_layer_call_and_return_conditional_losses_5316705o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
ї	
f
G__inference_dropout_01_layer_call_and_return_conditional_losses_5317000

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         2C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:М
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         2*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=ж
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         2o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         2i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         2Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         2:O K
'
_output_shapes
:         2
 
_user_specified_nameinputs
с
┐
D__inference_model_2_layer_call_and_return_conditional_losses_5316574

inputs*
fullyconnected0_5316520:	@╚&
fullyconnected0_5316522:	╚*
fullyconnected1_5316544:	╚2%
fullyconnected1_5316546:2 
output_5316568:2
output_5316570:
identityИв'FullyConnected0/StatefulPartitionedCallв'FullyConnected1/StatefulPartitionedCallвOutput/StatefulPartitionedCallУ
'FullyConnected0/StatefulPartitionedCallStatefulPartitionedCallinputsfullyconnected0_5316520fullyconnected0_5316522*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╚*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_FullyConnected0_layer_call_and_return_conditional_losses_5316519ы
dropout_00/PartitionedCallPartitionedCall0FullyConnected0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╚* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_dropout_00_layer_call_and_return_conditional_losses_5316530п
'FullyConnected1/StatefulPartitionedCallStatefulPartitionedCall#dropout_00/PartitionedCall:output:0fullyconnected1_5316544fullyconnected1_5316546*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_FullyConnected1_layer_call_and_return_conditional_losses_5316543ъ
dropout_01/PartitionedCallPartitionedCall0FullyConnected1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_dropout_01_layer_call_and_return_conditional_losses_5316554Л
Output/StatefulPartitionedCallStatefulPartitionedCall#dropout_01/PartitionedCall:output:0output_5316568output_5316570*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_Output_layer_call_and_return_conditional_losses_5316567v
IdentityIdentity'Output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ╗
NoOpNoOp(^FullyConnected0/StatefulPartitionedCall(^FullyConnected1/StatefulPartitionedCall^Output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : : : : : 2R
'FullyConnected0/StatefulPartitionedCall'FullyConnected0/StatefulPartitionedCall2R
'FullyConnected1/StatefulPartitionedCall'FullyConnected1/StatefulPartitionedCall2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
¤<
╙
 __inference__traced_save_5317124
file_prefix5
1savev2_fullyconnected0_kernel_read_readvariableop3
/savev2_fullyconnected0_bias_read_readvariableop5
1savev2_fullyconnected1_kernel_read_readvariableop3
/savev2_fullyconnected1_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop<
8savev2_adam_fullyconnected0_kernel_m_read_readvariableop:
6savev2_adam_fullyconnected0_bias_m_read_readvariableop<
8savev2_adam_fullyconnected1_kernel_m_read_readvariableop:
6savev2_adam_fullyconnected1_bias_m_read_readvariableop3
/savev2_adam_output_kernel_m_read_readvariableop1
-savev2_adam_output_bias_m_read_readvariableop<
8savev2_adam_fullyconnected0_kernel_v_read_readvariableop:
6savev2_adam_fullyconnected0_bias_v_read_readvariableop<
8savev2_adam_fullyconnected1_kernel_v_read_readvariableop:
6savev2_adam_fullyconnected1_bias_v_read_readvariableop3
/savev2_adam_output_kernel_v_read_readvariableop1
-savev2_adam_output_bias_v_read_readvariableop
savev2_const

identity_1ИвMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ї
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ю
valueФBСB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHе
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B ┼
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:01savev2_fullyconnected0_kernel_read_readvariableop/savev2_fullyconnected0_bias_read_readvariableop1savev2_fullyconnected1_kernel_read_readvariableop/savev2_fullyconnected1_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop8savev2_adam_fullyconnected0_kernel_m_read_readvariableop6savev2_adam_fullyconnected0_bias_m_read_readvariableop8savev2_adam_fullyconnected1_kernel_m_read_readvariableop6savev2_adam_fullyconnected1_bias_m_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableop8savev2_adam_fullyconnected0_kernel_v_read_readvariableop6savev2_adam_fullyconnected0_bias_v_read_readvariableop8savev2_adam_fullyconnected1_kernel_v_read_readvariableop6savev2_adam_fullyconnected1_bias_v_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 **
dtypes 
2	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*─
_input_shapes▓
п: :	@╚:╚:	╚2:2:2:: : : : : : : : : :	@╚:╚:	╚2:2:2::	@╚:╚:	╚2:2:2:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	@╚:!

_output_shapes	
:╚:%!

_output_shapes
:	╚2: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	@╚:!

_output_shapes	
:╚:%!

_output_shapes
:	╚2: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
::%!

_output_shapes
:	@╚:!

_output_shapes	
:╚:%!

_output_shapes
:	╚2: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
::

_output_shapes
: 
°
e
,__inference_dropout_01_layer_call_fn_5316983

inputs
identityИвStatefulPartitionedCall┼
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_dropout_01_layer_call_and_return_conditional_losses_5316619o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         222
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         2
 
_user_specified_nameinputs
┌
e
G__inference_dropout_01_layer_call_and_return_conditional_losses_5316988

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:         2[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         2"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         2:O K
'
_output_shapes
:         2
 
_user_specified_nameinputs
и

 
L__inference_FullyConnected0_layer_call_and_return_conditional_losses_5316519

inputs1
matmul_readvariableop_resource:	@╚.
biasadd_readvariableop_resource:	╚
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@╚*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚O
EluEluBiasAdd:output:0*
T0*(
_output_shapes
:         ╚a
IdentityIdentityElu:activations:0^NoOp*
T0*(
_output_shapes
:         ╚w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
╔/
╕
D__inference_model_2_layer_call_and_return_conditional_losses_5316906

inputsA
.fullyconnected0_matmul_readvariableop_resource:	@╚>
/fullyconnected0_biasadd_readvariableop_resource:	╚A
.fullyconnected1_matmul_readvariableop_resource:	╚2=
/fullyconnected1_biasadd_readvariableop_resource:27
%output_matmul_readvariableop_resource:24
&output_biasadd_readvariableop_resource:
identityИв&FullyConnected0/BiasAdd/ReadVariableOpв%FullyConnected0/MatMul/ReadVariableOpв&FullyConnected1/BiasAdd/ReadVariableOpв%FullyConnected1/MatMul/ReadVariableOpвOutput/BiasAdd/ReadVariableOpвOutput/MatMul/ReadVariableOpХ
%FullyConnected0/MatMul/ReadVariableOpReadVariableOp.fullyconnected0_matmul_readvariableop_resource*
_output_shapes
:	@╚*
dtype0К
FullyConnected0/MatMulMatMulinputs-FullyConnected0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚У
&FullyConnected0/BiasAdd/ReadVariableOpReadVariableOp/fullyconnected0_biasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype0з
FullyConnected0/BiasAddBiasAdd FullyConnected0/MatMul:product:0.FullyConnected0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚o
FullyConnected0/EluElu FullyConnected0/BiasAdd:output:0*
T0*(
_output_shapes
:         ╚]
dropout_00/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?Ц
dropout_00/dropout/MulMul!FullyConnected0/Elu:activations:0!dropout_00/dropout/Const:output:0*
T0*(
_output_shapes
:         ╚i
dropout_00/dropout/ShapeShape!FullyConnected0/Elu:activations:0*
T0*
_output_shapes
:г
/dropout_00/dropout/random_uniform/RandomUniformRandomUniform!dropout_00/dropout/Shape:output:0*
T0*(
_output_shapes
:         ╚*
dtype0f
!dropout_00/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=╚
dropout_00/dropout/GreaterEqualGreaterEqual8dropout_00/dropout/random_uniform/RandomUniform:output:0*dropout_00/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ╚Ж
dropout_00/dropout/CastCast#dropout_00/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ╚Л
dropout_00/dropout/Mul_1Muldropout_00/dropout/Mul:z:0dropout_00/dropout/Cast:y:0*
T0*(
_output_shapes
:         ╚Х
%FullyConnected1/MatMul/ReadVariableOpReadVariableOp.fullyconnected1_matmul_readvariableop_resource*
_output_shapes
:	╚2*
dtype0Я
FullyConnected1/MatMulMatMuldropout_00/dropout/Mul_1:z:0-FullyConnected1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2Т
&FullyConnected1/BiasAdd/ReadVariableOpReadVariableOp/fullyconnected1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0ж
FullyConnected1/BiasAddBiasAdd FullyConnected1/MatMul:product:0.FullyConnected1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2n
FullyConnected1/EluElu FullyConnected1/BiasAdd:output:0*
T0*'
_output_shapes
:         2]
dropout_01/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?Х
dropout_01/dropout/MulMul!FullyConnected1/Elu:activations:0!dropout_01/dropout/Const:output:0*
T0*'
_output_shapes
:         2i
dropout_01/dropout/ShapeShape!FullyConnected1/Elu:activations:0*
T0*
_output_shapes
:в
/dropout_01/dropout/random_uniform/RandomUniformRandomUniform!dropout_01/dropout/Shape:output:0*
T0*'
_output_shapes
:         2*
dtype0f
!dropout_01/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=╟
dropout_01/dropout/GreaterEqualGreaterEqual8dropout_01/dropout/random_uniform/RandomUniform:output:0*dropout_01/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         2Е
dropout_01/dropout/CastCast#dropout_01/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         2К
dropout_01/dropout/Mul_1Muldropout_01/dropout/Mul:z:0dropout_01/dropout/Cast:y:0*
T0*'
_output_shapes
:         2В
Output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0Н
Output/MatMulMatMuldropout_01/dropout/Mul_1:z:0$Output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         А
Output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Л
Output/BiasAddBiasAddOutput/MatMul:product:0%Output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d
Output/SoftmaxSoftmaxOutput/BiasAdd:output:0*
T0*'
_output_shapes
:         g
IdentityIdentityOutput/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         з
NoOpNoOp'^FullyConnected0/BiasAdd/ReadVariableOp&^FullyConnected0/MatMul/ReadVariableOp'^FullyConnected1/BiasAdd/ReadVariableOp&^FullyConnected1/MatMul/ReadVariableOp^Output/BiasAdd/ReadVariableOp^Output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : : : : : 2P
&FullyConnected0/BiasAdd/ReadVariableOp&FullyConnected0/BiasAdd/ReadVariableOp2N
%FullyConnected0/MatMul/ReadVariableOp%FullyConnected0/MatMul/ReadVariableOp2P
&FullyConnected1/BiasAdd/ReadVariableOp&FullyConnected1/BiasAdd/ReadVariableOp2N
%FullyConnected1/MatMul/ReadVariableOp%FullyConnected1/MatMul/ReadVariableOp2>
Output/BiasAdd/ReadVariableOpOutput/BiasAdd/ReadVariableOp2<
Output/MatMul/ReadVariableOpOutput/MatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
№
e
,__inference_dropout_00_layer_call_fn_5316936

inputs
identityИвStatefulPartitionedCall╞
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╚* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_dropout_00_layer_call_and_return_conditional_losses_5316652p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ╚`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ╚22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs
ї	
f
G__inference_dropout_01_layer_call_and_return_conditional_losses_5316619

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         2C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:М
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         2*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=ж
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         2o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         2i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         2Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         2:O K
'
_output_shapes
:         2
 
_user_specified_nameinputs
д

■
L__inference_FullyConnected1_layer_call_and_return_conditional_losses_5316973

inputs1
matmul_readvariableop_resource:	╚2-
biasadd_readvariableop_resource:2
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	╚2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2N
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:         2`
IdentityIdentityElu:activations:0^NoOp*
T0*'
_output_shapes
:         2w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ╚: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs
█
И
D__inference_model_2_layer_call_and_return_conditional_losses_5316779	
input*
fullyconnected0_5316761:	@╚&
fullyconnected0_5316763:	╚*
fullyconnected1_5316767:	╚2%
fullyconnected1_5316769:2 
output_5316773:2
output_5316775:
identityИв'FullyConnected0/StatefulPartitionedCallв'FullyConnected1/StatefulPartitionedCallвOutput/StatefulPartitionedCallв"dropout_00/StatefulPartitionedCallв"dropout_01/StatefulPartitionedCallТ
'FullyConnected0/StatefulPartitionedCallStatefulPartitionedCallinputfullyconnected0_5316761fullyconnected0_5316763*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╚*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_FullyConnected0_layer_call_and_return_conditional_losses_5316519√
"dropout_00/StatefulPartitionedCallStatefulPartitionedCall0FullyConnected0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╚* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_dropout_00_layer_call_and_return_conditional_losses_5316652╖
'FullyConnected1/StatefulPartitionedCallStatefulPartitionedCall+dropout_00/StatefulPartitionedCall:output:0fullyconnected1_5316767fullyconnected1_5316769*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_FullyConnected1_layer_call_and_return_conditional_losses_5316543Я
"dropout_01/StatefulPartitionedCallStatefulPartitionedCall0FullyConnected1/StatefulPartitionedCall:output:0#^dropout_00/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_dropout_01_layer_call_and_return_conditional_losses_5316619У
Output/StatefulPartitionedCallStatefulPartitionedCall+dropout_01/StatefulPartitionedCall:output:0output_5316773output_5316775*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_Output_layer_call_and_return_conditional_losses_5316567v
IdentityIdentity'Output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Е
NoOpNoOp(^FullyConnected0/StatefulPartitionedCall(^FullyConnected1/StatefulPartitionedCall^Output/StatefulPartitionedCall#^dropout_00/StatefulPartitionedCall#^dropout_01/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : : : : : 2R
'FullyConnected0/StatefulPartitionedCall'FullyConnected0/StatefulPartitionedCall2R
'FullyConnected1/StatefulPartitionedCall'FullyConnected1/StatefulPartitionedCall2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall2H
"dropout_00/StatefulPartitionedCall"dropout_00/StatefulPartitionedCall2H
"dropout_01/StatefulPartitionedCall"dropout_01/StatefulPartitionedCall:N J
'
_output_shapes
:         @

_user_specified_nameinput
ж
H
,__inference_dropout_01_layer_call_fn_5316978

inputs
identity╡
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_dropout_01_layer_call_and_return_conditional_losses_5316554`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         2:O K
'
_output_shapes
:         2
 
_user_specified_nameinputs
├
Х
(__inference_Output_layer_call_fn_5317009

inputs
unknown:2
	unknown_0:
identityИвStatefulPartitionedCall█
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_Output_layer_call_and_return_conditional_losses_5316567o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         2: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         2
 
_user_specified_nameinputs
┌
e
G__inference_dropout_01_layer_call_and_return_conditional_losses_5316554

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:         2[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         2"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         2:O K
'
_output_shapes
:         2
 
_user_specified_nameinputs
к
H
,__inference_dropout_00_layer_call_fn_5316931

inputs
identity╢
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╚* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_dropout_00_layer_call_and_return_conditional_losses_5316530a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ╚"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ╚:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs
┘
а
1__inference_FullyConnected0_layer_call_fn_5316915

inputs
unknown:	@╚
	unknown_0:	╚
identityИвStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╚*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_FullyConnected0_layer_call_and_return_conditional_losses_5316519p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ╚`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
Я

Ї
C__inference_Output_layer_call_and_return_conditional_losses_5317020

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         2
 
_user_specified_nameinputs
уm
─
#__inference__traced_restore_5317215
file_prefix:
'assignvariableop_fullyconnected0_kernel:	@╚6
'assignvariableop_1_fullyconnected0_bias:	╚<
)assignvariableop_2_fullyconnected1_kernel:	╚25
'assignvariableop_3_fullyconnected1_bias:22
 assignvariableop_4_output_kernel:2,
assignvariableop_5_output_bias:&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: %
assignvariableop_11_total_1: %
assignvariableop_12_count_1: #
assignvariableop_13_total: #
assignvariableop_14_count: D
1assignvariableop_15_adam_fullyconnected0_kernel_m:	@╚>
/assignvariableop_16_adam_fullyconnected0_bias_m:	╚D
1assignvariableop_17_adam_fullyconnected1_kernel_m:	╚2=
/assignvariableop_18_adam_fullyconnected1_bias_m:2:
(assignvariableop_19_adam_output_kernel_m:24
&assignvariableop_20_adam_output_bias_m:D
1assignvariableop_21_adam_fullyconnected0_kernel_v:	@╚>
/assignvariableop_22_adam_fullyconnected0_bias_v:	╚D
1assignvariableop_23_adam_fullyconnected1_kernel_v:	╚2=
/assignvariableop_24_adam_fullyconnected1_bias_v:2:
(assignvariableop_25_adam_output_kernel_v:24
&assignvariableop_26_adam_output_bias_v:
identity_28ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9°
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ю
valueФBСB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHи
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B л
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Д
_output_shapesr
p::::::::::::::::::::::::::::**
dtypes 
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOpAssignVariableOp'assignvariableop_fullyconnected0_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_1AssignVariableOp'assignvariableop_1_fullyconnected0_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_2AssignVariableOp)assignvariableop_2_fullyconnected1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_3AssignVariableOp'assignvariableop_3_fullyconnected1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_4AssignVariableOp assignvariableop_4_output_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_5AssignVariableOpassignvariableop_5_output_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:Л
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_11AssignVariableOpassignvariableop_11_total_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_12AssignVariableOpassignvariableop_12_count_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:в
AssignVariableOp_15AssignVariableOp1assignvariableop_15_adam_fullyconnected0_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_16AssignVariableOp/assignvariableop_16_adam_fullyconnected0_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:в
AssignVariableOp_17AssignVariableOp1assignvariableop_17_adam_fullyconnected1_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_18AssignVariableOp/assignvariableop_18_adam_fullyconnected1_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_output_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_output_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:в
AssignVariableOp_21AssignVariableOp1assignvariableop_21_adam_fullyconnected0_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_22AssignVariableOp/assignvariableop_22_adam_fullyconnected0_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:в
AssignVariableOp_23AssignVariableOp1assignvariableop_23_adam_fullyconnected1_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_24AssignVariableOp/assignvariableop_24_adam_fullyconnected1_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_output_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_26AssignVariableOp&assignvariableop_26_adam_output_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 б
Identity_27Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_28IdentityIdentity_27:output:0^NoOp_1*
T0*
_output_shapes
: О
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_28Identity_28:output:0*K
_input_shapes:
8: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
¤	
f
G__inference_dropout_00_layer_call_and_return_conditional_losses_5316953

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         ╚C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         ╚*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=з
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ╚p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ╚j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         ╚Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         ╚"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ╚:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs
Ё
Е
)__inference_model_2_layer_call_fn_5316821

inputs
unknown:	@╚
	unknown_0:	╚
	unknown_1:	╚2
	unknown_2:2
	unknown_3:2
	unknown_4:
identityИвStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_model_2_layer_call_and_return_conditional_losses_5316574o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
▐
e
G__inference_dropout_00_layer_call_and_return_conditional_losses_5316941

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         ╚\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ╚"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ╚:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs"┐L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*е
serving_defaultС
7
input.
serving_default_input:0         @:
Output0
StatefulPartitionedCall:0         tensorflow/serving/predict:за
 
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
╗
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
╝
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator"
_tf_keras_layer
╗
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

%kernel
&bias"
_tf_keras_layer
╝
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses
-_random_generator"
_tf_keras_layer
╗
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses

4kernel
5bias"
_tf_keras_layer
J
0
1
%2
&3
44
55"
trackable_list_wrapper
J
0
1
%2
&3
44
55"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
6non_trainable_variables

7layers
8metrics
9layer_regularization_losses
:layer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
┌
;trace_0
<trace_1
=trace_2
>trace_32я
)__inference_model_2_layer_call_fn_5316589
)__inference_model_2_layer_call_fn_5316821
)__inference_model_2_layer_call_fn_5316838
)__inference_model_2_layer_call_fn_5316737└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 z;trace_0z<trace_1z=trace_2z>trace_3
╞
?trace_0
@trace_1
Atrace_2
Btrace_32█
D__inference_model_2_layer_call_and_return_conditional_losses_5316865
D__inference_model_2_layer_call_and_return_conditional_losses_5316906
D__inference_model_2_layer_call_and_return_conditional_losses_5316758
D__inference_model_2_layer_call_and_return_conditional_losses_5316779└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 z?trace_0z@trace_1zAtrace_2zBtrace_3
╦B╚
"__inference__wrapped_model_5316501input"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╞
Citer

Dbeta_1

Ebeta_2
	Fdecay
Glearning_ratem{m|%m}&m~4m5mАvБvВ%vГ&vД4vЕ5vЖ"
	optimizer
,
Hserving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
н
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ї
Ntrace_02╪
1__inference_FullyConnected0_layer_call_fn_5316915в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zNtrace_0
Р
Otrace_02є
L__inference_FullyConnected0_layer_call_and_return_conditional_losses_5316926в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zOtrace_0
):'	@╚2FullyConnected0/kernel
#:!╚2FullyConnected0/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
╩
Utrace_0
Vtrace_12У
,__inference_dropout_00_layer_call_fn_5316931
,__inference_dropout_00_layer_call_fn_5316936┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 zUtrace_0zVtrace_1
А
Wtrace_0
Xtrace_12╔
G__inference_dropout_00_layer_call_and_return_conditional_losses_5316941
G__inference_dropout_00_layer_call_and_return_conditional_losses_5316953┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 zWtrace_0zXtrace_1
"
_generic_user_object
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
н
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
ї
^trace_02╪
1__inference_FullyConnected1_layer_call_fn_5316962в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z^trace_0
Р
_trace_02є
L__inference_FullyConnected1_layer_call_and_return_conditional_losses_5316973в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z_trace_0
):'	╚22FullyConnected1/kernel
": 22FullyConnected1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
╩
etrace_0
ftrace_12У
,__inference_dropout_01_layer_call_fn_5316978
,__inference_dropout_01_layer_call_fn_5316983┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 zetrace_0zftrace_1
А
gtrace_0
htrace_12╔
G__inference_dropout_01_layer_call_and_return_conditional_losses_5316988
G__inference_dropout_01_layer_call_and_return_conditional_losses_5317000┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 zgtrace_0zhtrace_1
"
_generic_user_object
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
н
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
ь
ntrace_02╧
(__inference_Output_layer_call_fn_5317009в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zntrace_0
З
otrace_02ъ
C__inference_Output_layer_call_and_return_conditional_losses_5317020в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zotrace_0
:22Output/kernel
:2Output/bias
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
·Bў
)__inference_model_2_layer_call_fn_5316589input"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
√B°
)__inference_model_2_layer_call_fn_5316821inputs"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
√B°
)__inference_model_2_layer_call_fn_5316838inputs"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
·Bў
)__inference_model_2_layer_call_fn_5316737input"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ЦBУ
D__inference_model_2_layer_call_and_return_conditional_losses_5316865inputs"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ЦBУ
D__inference_model_2_layer_call_and_return_conditional_losses_5316906inputs"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ХBТ
D__inference_model_2_layer_call_and_return_conditional_losses_5316758input"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ХBТ
D__inference_model_2_layer_call_and_return_conditional_losses_5316779input"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
╩B╟
%__inference_signature_wrapper_5316804input"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
хBт
1__inference_FullyConnected0_layer_call_fn_5316915inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
АB¤
L__inference_FullyConnected0_layer_call_and_return_conditional_losses_5316926inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЄBя
,__inference_dropout_00_layer_call_fn_5316931inputs"┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ЄBя
,__inference_dropout_00_layer_call_fn_5316936inputs"┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
НBК
G__inference_dropout_00_layer_call_and_return_conditional_losses_5316941inputs"┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
НBК
G__inference_dropout_00_layer_call_and_return_conditional_losses_5316953inputs"┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
хBт
1__inference_FullyConnected1_layer_call_fn_5316962inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
АB¤
L__inference_FullyConnected1_layer_call_and_return_conditional_losses_5316973inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЄBя
,__inference_dropout_01_layer_call_fn_5316978inputs"┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ЄBя
,__inference_dropout_01_layer_call_fn_5316983inputs"┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
НBК
G__inference_dropout_01_layer_call_and_return_conditional_losses_5316988inputs"┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
НBК
G__inference_dropout_01_layer_call_and_return_conditional_losses_5317000inputs"┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▄B┘
(__inference_Output_layer_call_fn_5317009inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
C__inference_Output_layer_call_and_return_conditional_losses_5317020inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
N
r	variables
s	keras_api
	ttotal
	ucount"
_tf_keras_metric
^
v	variables
w	keras_api
	xtotal
	ycount
z
_fn_kwargs"
_tf_keras_metric
.
t0
u1"
trackable_list_wrapper
-
r	variables"
_generic_user_object
:  (2total
:  (2count
.
x0
y1"
trackable_list_wrapper
-
v	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.:,	@╚2Adam/FullyConnected0/kernel/m
(:&╚2Adam/FullyConnected0/bias/m
.:,	╚22Adam/FullyConnected1/kernel/m
':%22Adam/FullyConnected1/bias/m
$:"22Adam/Output/kernel/m
:2Adam/Output/bias/m
.:,	@╚2Adam/FullyConnected0/kernel/v
(:&╚2Adam/FullyConnected0/bias/v
.:,	╚22Adam/FullyConnected1/kernel/v
':%22Adam/FullyConnected1/bias/v
$:"22Adam/Output/kernel/v
:2Adam/Output/bias/vн
L__inference_FullyConnected0_layer_call_and_return_conditional_losses_5316926]/в,
%в"
 К
inputs         @
к "&в#
К
0         ╚
Ъ Е
1__inference_FullyConnected0_layer_call_fn_5316915P/в,
%в"
 К
inputs         @
к "К         ╚н
L__inference_FullyConnected1_layer_call_and_return_conditional_losses_5316973]%&0в-
&в#
!К
inputs         ╚
к "%в"
К
0         2
Ъ Е
1__inference_FullyConnected1_layer_call_fn_5316962P%&0в-
&в#
!К
inputs         ╚
к "К         2г
C__inference_Output_layer_call_and_return_conditional_losses_5317020\45/в,
%в"
 К
inputs         2
к "%в"
К
0         
Ъ {
(__inference_Output_layer_call_fn_5317009O45/в,
%в"
 К
inputs         2
к "К         П
"__inference__wrapped_model_5316501i%&45.в+
$в!
К
input         @
к "/к,
*
Output К
Output         й
G__inference_dropout_00_layer_call_and_return_conditional_losses_5316941^4в1
*в'
!К
inputs         ╚
p 
к "&в#
К
0         ╚
Ъ й
G__inference_dropout_00_layer_call_and_return_conditional_losses_5316953^4в1
*в'
!К
inputs         ╚
p
к "&в#
К
0         ╚
Ъ Б
,__inference_dropout_00_layer_call_fn_5316931Q4в1
*в'
!К
inputs         ╚
p 
к "К         ╚Б
,__inference_dropout_00_layer_call_fn_5316936Q4в1
*в'
!К
inputs         ╚
p
к "К         ╚з
G__inference_dropout_01_layer_call_and_return_conditional_losses_5316988\3в0
)в&
 К
inputs         2
p 
к "%в"
К
0         2
Ъ з
G__inference_dropout_01_layer_call_and_return_conditional_losses_5317000\3в0
)в&
 К
inputs         2
p
к "%в"
К
0         2
Ъ 
,__inference_dropout_01_layer_call_fn_5316978O3в0
)в&
 К
inputs         2
p 
к "К         2
,__inference_dropout_01_layer_call_fn_5316983O3в0
)в&
 К
inputs         2
p
к "К         2п
D__inference_model_2_layer_call_and_return_conditional_losses_5316758g%&456в3
,в)
К
input         @
p 

 
к "%в"
К
0         
Ъ п
D__inference_model_2_layer_call_and_return_conditional_losses_5316779g%&456в3
,в)
К
input         @
p

 
к "%в"
К
0         
Ъ ░
D__inference_model_2_layer_call_and_return_conditional_losses_5316865h%&457в4
-в*
 К
inputs         @
p 

 
к "%в"
К
0         
Ъ ░
D__inference_model_2_layer_call_and_return_conditional_losses_5316906h%&457в4
-в*
 К
inputs         @
p

 
к "%в"
К
0         
Ъ З
)__inference_model_2_layer_call_fn_5316589Z%&456в3
,в)
К
input         @
p 

 
к "К         З
)__inference_model_2_layer_call_fn_5316737Z%&456в3
,в)
К
input         @
p

 
к "К         И
)__inference_model_2_layer_call_fn_5316821[%&457в4
-в*
 К
inputs         @
p 

 
к "К         И
)__inference_model_2_layer_call_fn_5316838[%&457в4
-в*
 К
inputs         @
p

 
к "К         Ы
%__inference_signature_wrapper_5316804r%&457в4
в 
-к*
(
inputК
input         @"/к,
*
Output К
Output         