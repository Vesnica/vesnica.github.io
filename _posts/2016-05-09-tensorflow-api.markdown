---
layout: post
title:  "TensorFlow API List"
date:   2016-05-09 19:12:08 +0800
categories: tensorflow
permalink: tensorflow_api
---

###  Building Graphs

#### Core graph data structures
*       tf.Graph
*       tf.Operation
*       tf.Tensor

#### Tensor types
*       tf.DType
*       tf.as_dtype()

#### Utility functions
*       tf.device()
*       tf.name_scope()
*       tf.control_dependencies()
*       tf.convert_to_tensor()
*       tf.convert_to_tensor_or_indexed_slices()
*       tf.get_default_graph()
*       tf.reset_default_graph()
*       tf.import_graph_def()
*       tf.load_file_system_library()
*       tf.load_op_library()

#### Graph collections
*       tf.add_to_collection()
*       tf.get_collection()
*       tf.get_collection_ref()
*       tf.GraphKeys

#### Defining new operations
*       tf.RegisterGradient
*       tf.NoGradient()
*       tf.RegisterShape
*       tf.TensorShape
*       tf.Dimension
*       tf.op_scope()
*       tf.get_seed()

#### For libraries building on TensorFlow
*       tf.register_tensor_conversion_function()

#### Other Functions and Classes
*       tf.DeviceSpec
*       tf.bytes

###  Control Flow

#### Control Flow Operations
*       tf.identity()
*       tf.tuple()
*       tf.group()
*       tf.no_op()
*       tf.count_up_to()
*       tf.cond()
*       tf.case()
*       tf.while_loop()

#### Logical Operators
*       tf.logical_and()
*       tf.logical_not()
*       tf.logical_or()
*       tf.logical_xor()

#### Comparison Operators
*       tf.equal()
*       tf.not_equal()
*       tf.less()
*       tf.less_equal()
*       tf.greater()
*       tf.greater_equal()
*       tf.select()
*       tf.where()

#### Debugging Operations
*       tf.is_finite()
*       tf.is_inf()
*       tf.is_nan()
*       tf.verify_tensor_all_finite()
*       tf.check_numerics()
*       tf.add_check_numerics_ops()
*       tf.Assert()
*       tf.Print()

###  Running Graphs

#### Session management
*       tf.Session
*       tf.InteractiveSession
*       tf.get_default_session()

#### Error classes
*       tf.OpError
*       tf.errors.CancelledError
*       tf.errors.UnknownError
*       tf.errors.InvalidArgumentError
*       tf.errors.DeadlineExceededError
*       tf.errors.NotFoundError
*       tf.errors.AlreadyExistsError
*       tf.errors.PermissionDeniedError
*       tf.errors.UnauthenticatedError
*       tf.errors.ResourceExhaustedError
*       tf.errors.FailedPreconditionError
*       tf.errors.AbortedError
*       tf.errors.OutOfRangeError
*       tf.errors.UnimplementedError
*       tf.errors.InternalError
*       tf.errors.UnavailableError
*       tf.errors.DataLossError

###  Training

#### Optimizers
*       tf.train.Optimizer
*       tf.train.GradientDescentOptimizer
*       tf.train.AdadeltaOptimizer
*       tf.train.AdagradOptimizer
*       tf.train.MomentumOptimizer
*       tf.train.AdamOptimizer
*       tf.train.FtrlOptimizer
*       tf.train.RMSPropOptimizer

#### Gradient Computation
*       tf.gradients()
*       tf.AggregationMethod
*       tf.stop_gradient()

#### Gradient Clipping
*       tf.clip_by_value()
*       tf.clip_by_norm()
*       tf.clip_by_average_norm()
*       tf.clip_by_global_norm()
*       tf.global_norm()

#### Decaying the learning rate
*       tf.train.exponential_decay()

#### Moving Averages
*       tf.train.ExponentialMovingAverage

#### Coordinator and QueueRunner
*       tf.train.Coordinator
*       tf.train.QueueRunner
*       tf.train.add_queue_runner()
*       tf.train.start_queue_runners()

#### Distributed execution
*       tf.train.Server
*       tf.train.Supervisor
*       tf.train.SessionManager
*       tf.train.ClusterSpec
*       tf.train.replica_device_setter()

#### Summary Operations
*       tf.scalar_summary()
*       tf.image_summary()
*       tf.audio_summary()
*       tf.histogram_summary()
*       tf.nn.zero_fraction()
*       tf.merge_summary()
*       tf.merge_all_summaries()

#### Adding Summaries to Event Files
*       tf.train.SummaryWriter
*       tf.train.summary_iterator()

#### Training utilities
*       tf.train.global_step()
*       tf.train.write_graph()

#### Other Functions and Classes
*       tf.train.LooperThread
*       tf.train.generate_checkpoint_state_proto()

###  Neural Network

#### Activation Functions
*       tf.nn.relu()
*       tf.nn.relu6()
*       tf.nn.elu()
*       tf.nn.softplus()
*       tf.nn.softsign()
*       tf.nn.dropout()
*       tf.nn.bias_add()
*       tf.sigmoid()
*       tf.tanh()

#### Convolution
*       tf.nn.conv2d()
*       tf.nn.depthwise_conv2d()
*       tf.nn.separable_conv2d()
*       tf.nn.atrous_conv2d()
*       tf.nn.conv2d_transpose()

#### Pooling
*       tf.nn.avg_pool()
*       tf.nn.max_pool()
*       tf.nn.max_pool_with_argmax()

#### Normalization
*       tf.nn.l2_normalize()
*       tf.nn.local_response_normalization()
*       tf.nn.sufficient_statistics()
*       tf.nn.normalize_moments()
*       tf.nn.moments()

#### Losses
*       tf.nn.l2_loss()

#### Classification
*       tf.nn.sigmoid_cross_entropy_with_logits()
*       tf.nn.softmax()
*       tf.nn.log_softmax()
*       tf.nn.softmax_cross_entropy_with_logits()
*       tf.nn.sparse_softmax_cross_entropy_with_logits()
*       tf.nn.weighted_cross_entropy_with_logits()

#### Embeddings
*       tf.nn.embedding_lookup()
*       tf.nn.embedding_lookup_sparse()

#### Evaluation
*       tf.nn.top_k()
*       tf.nn.in_top_k()

#### Candidate Sampling
*       tf.nn.nce_loss()
*       tf.nn.sampled_softmax_loss()
*       tf.nn.uniform_candidate_sampler()
*       tf.nn.log_uniform_candidate_sampler()
*       tf.nn.learned_unigram_candidate_sampler()
*       tf.nn.fixed_unigram_candidate_sampler()

#### Miscellaneous candidate sampling utilities
*       tf.nn.compute_accidental_hits()

#### Other Functions and Classes
*       tf.nn.batch_normalization()
*       tf.nn.depthwise_conv2d_native()

###  Constants, Sequences, and Random Values

#### Constant Value Tensors
*       tf.zeros()
*       tf.zeros_like()
*       tf.ones()
*       tf.ones_like()
*       tf.fill()
*       tf.constant()

#### Sequences
*       tf.linspace()
*       tf.range()

#### Random Tensors
*       tf.random_normal()
*       tf.truncated_normal()
*       tf.random_uniform()
*       tf.random_shuffle()
*       tf.random_crop()
*       tf.set_random_seed()

###  Variables

#### Variables
*       tf.Variable

#### Variable helper functions
*       tf.all_variables()
*       tf.trainable_variables()
*       tf.local_variables()
*       tf.moving_average_variables()
*       tf.initialize_all_variables()
*       tf.initialize_variables()
*       tf.initialize_local_variables()
*       tf.is_variable_initialized()
*       tf.assert_variables_initialized()

#### Saving and Restoring Variables
*       tf.train.Saver
*       tf.train.latest_checkpoint()
*       tf.train.get_checkpoint_state()
*       tf.train.update_checkpoint_state()

#### Sharing Variables
*       tf.get_variable()
*       tf.VariableScope
*       tf.variable_scope()
*       tf.variable_op_scope()
*       tf.get_variable_scope()
*       tf.make_template()
*       tf.no_regularizer()
*       tf.constant_initializer()
*       tf.random_normal_initializer()
*       tf.truncated_normal_initializer()
*       tf.random_uniform_initializer()
*       tf.uniform_unit_scaling_initializer()
*       tf.zeros_initializer()
*       tf.ones_initializer()

#### Variable Partitioners for Sharding
*       tf.variable_axis_size_partitioner()

#### Sparse Variable Updates
*       tf.scatter_update()
*       tf.scatter_add()
*       tf.scatter_sub()
*       tf.sparse_mask()
*       tf.IndexedSlices

#### Exporting and Importing Meta Graphs
*       tf.train.export_meta_graph()
*       tf.train.import_meta_graph()

###  Tensor Transformations

#### Casting
*       tf.string_to_number()
*       tf.to_double()
*       tf.to_float()
*       tf.to_bfloat16()
*       tf.to_int32()
*       tf.to_int64()
*       tf.cast()
*       tf.saturate_cast()

#### Shapes and Shaping
*       tf.shape()
*       tf.size()
*       tf.rank()
*       tf.reshape()
*       tf.squeeze()
*       tf.expand_dims()

#### Slicing and Joining
*       tf.slice()
*       tf.split()
*       tf.tile()
*       tf.pad()
*       tf.concat()
*       tf.pack()
*       tf.unpack()
*       tf.reverse_sequence()
*       tf.reverse()
*       tf.transpose()
*       tf.space_to_batch()
*       tf.batch_to_space()
*       tf.space_to_depth()
*       tf.depth_to_space()
*       tf.gather()
*       tf.gather_nd()
*       tf.dynamic_partition()
*       tf.dynamic_stitch()
*       tf.boolean_mask()
*       tf.one_hot()

#### Other Functions and Classes
*       tf.bitcast()
*       tf.shape_n()
*       tf.unique_with_counts()

###  Sparse Tensors

#### Sparse Tensor Representation
*       tf.SparseTensor
*       tf.SparseTensorValue

#### Conversion
*       tf.sparse_to_dense()
*       tf.sparse_tensor_to_dense()
*       tf.sparse_to_indicator()
*       tf.sparse_merge()

#### Manipulation
*       tf.sparse_concat()
*       tf.sparse_reorder()
*       tf.sparse_split()
*       tf.sparse_retain()
*       tf.sparse_fill_empty_rows()

#### Math Operations
*       tf.sparse_add()
*       tf.sparse_tensor_dense_matmul()

###  Math

#### Arithmetic Operators
*       tf.add()
*       tf.sub()
*       tf.mul()
*       tf.div()
*       tf.truediv()
*       tf.floordiv()
*       tf.mod()
*       tf.cross()

#### Basic Math Functions
*       tf.add_n()
*       tf.abs()
*       tf.neg()
*       tf.sign()
*       tf.inv()
*       tf.square()
*       tf.round()
*       tf.sqrt()
*       tf.rsqrt()
*       tf.pow()
*       tf.exp()
*       tf.log()
*       tf.ceil()
*       tf.floor()
*       tf.maximum()
*       tf.minimum()
*       tf.cos()
*       tf.sin()
*       tf.lbeta()
*       tf.lgamma()
*       tf.digamma()
*       tf.erf()
*       tf.erfc()
*       tf.squared_difference()
*       tf.igamma()
*       tf.igammac()

#### Matrix Math Functions
*       tf.batch_matrix_diag()
*       tf.batch_matrix_diag_part()
*       tf.batch_matrix_band_part()
*       tf.diag()
*       tf.diag_part()
*       tf.trace()
*       tf.transpose()
*       tf.matmul()
*       tf.batch_matmul()
*       tf.matrix_determinant()
*       tf.batch_matrix_determinant()
*       tf.matrix_inverse()
*       tf.batch_matrix_inverse()
*       tf.cholesky()
*       tf.batch_cholesky()
*       tf.self_adjoint_eig()
*       tf.batch_self_adjoint_eig()
*       tf.matrix_solve()
*       tf.batch_matrix_solve()
*       tf.matrix_triangular_solve()
*       tf.batch_matrix_triangular_solve()
*       tf.matrix_solve_ls()
*       tf.batch_matrix_solve_ls()

#### Complex Number Functions
*       tf.complex()
*       tf.complex_abs()
*       tf.conj()
*       tf.imag()
*       tf.real()
*       tf.fft()
*       tf.ifft()
*       tf.fft2d()
*       tf.ifft2d()
*       tf.fft3d()
*       tf.ifft3d()
*       tf.batch_fft()
*       tf.batch_ifft()
*       tf.batch_fft2d()
*       tf.batch_ifft2d()
*       tf.batch_fft3d()
*       tf.batch_ifft3d()

#### Reduction
*       tf.reduce_sum()
*       tf.reduce_prod()
*       tf.reduce_min()
*       tf.reduce_max()
*       tf.reduce_mean()
*       tf.reduce_all()
*       tf.reduce_any()
*       tf.accumulate_n()

#### Segmentation
*       tf.segment_sum()
*       tf.segment_prod()
*       tf.segment_min()
*       tf.segment_max()
*       tf.segment_mean()
*       tf.unsorted_segment_sum()
*       tf.sparse_segment_sum()
*       tf.sparse_segment_mean()
*       tf.sparse_segment_sqrt_n()

#### Sequence Comparison and Indexing
*       tf.argmin()
*       tf.argmax()
*       tf.listdiff()
*       tf.where()
*       tf.unique()
*       tf.edit_distance()
*       tf.invert_permutation()

#### Other Functions and Classes
*       tf.scalar_mul()
*       tf.sparse_segment_sqrt_n_grad()

###  Inputs and Readers

#### Placeholders
*       tf.placeholder()
*       tf.placeholder_with_default()

#### Readers
*       tf.ReaderBase
*       tf.TextLineReader
*       tf.WholeFileReader
*       tf.IdentityReader
*       tf.TFRecordReader
*       tf.FixedLengthRecordReader

#### Converting
*       tf.decode_csv()
*       tf.decode_raw()

#### Example protocol buffer
*       tf.VarLenFeature
*       tf.FixedLenFeature
*       tf.FixedLenSequenceFeature
*       tf.parse_example()
*       tf.parse_single_example()
*       tf.decode_json_example()

#### Queues
*       tf.QueueBase
*       tf.FIFOQueue
*       tf.RandomShuffleQueue

#### Dealing with the filesystem
*       tf.matching_files()
*       tf.read_file()

#### Input pipeline
*       tf.train.match_filenames_once()
*       tf.train.limit_epochs()
*       tf.train.input_producer()
*       tf.train.range_input_producer()
*       tf.train.slice_input_producer()
*       tf.train.string_input_producer()

#### Batching at the end of an input pipeline
*       tf.train.batch()
*       tf.train.batch_join()
*       tf.train.shuffle_batch()
*       tf.train.shuffle_batch_join()

###  Data IO (Python functions)

#### Data IO (Python Functions)
*       tf.python_io.TFRecordWriter

###  Images

#### Encoding and Decoding
*       tf.image.decode_jpeg()
*       tf.image.encode_jpeg()
*       tf.image.decode_png()
*       tf.image.encode_png()

#### Resizing
*       tf.image.resize_images()
*       tf.image.resize_area()
*       tf.image.resize_bicubic()
*       tf.image.resize_bilinear()
*       tf.image.resize_nearest_neighbor()

#### Cropping
*       tf.image.resize_image_with_crop_or_pad()
*       tf.image.central_crop()
*       tf.image.pad_to_bounding_box()
*       tf.image.crop_to_bounding_box()
*       tf.image.extract_glimpse()

#### Flipping and Transposing
*       tf.image.flip_up_down()
*       tf.image.random_flip_up_down()
*       tf.image.flip_left_right()
*       tf.image.random_flip_left_right()
*       tf.image.transpose_image()

#### Converting Between Colorspaces
*       tf.image.rgb_to_grayscale()
*       tf.image.grayscale_to_rgb()
*       tf.image.hsv_to_rgb()
*       tf.image.rgb_to_hsv()
*       tf.image.convert_image_dtype()

#### Image Adjustments
*       tf.image.adjust_brightness()
*       tf.image.random_brightness()
*       tf.image.adjust_contrast()
*       tf.image.random_contrast()
*       tf.image.adjust_hue()
*       tf.image.random_hue()
*       tf.image.adjust_saturation()
*       tf.image.random_saturation()
*       tf.image.per_image_whitening()

#### Working with Bounding Boxes
*       tf.image.draw_bounding_boxes()
*       tf.image.sample_distorted_bounding_box()

###  Testing

#### Unit tests
*       tf.test.main()

#### Utilities
*       tf.test.assert_equal_graph_def()
*       tf.test.get_temp_dir()
*       tf.test.is_built_with_cuda()

#### Gradient checking
*       tf.test.compute_gradient()
*       tf.test.compute_gradient_error()

###  Layers (contrib)

#### Higher level ops for building neural network layers
*       tf.contrib.layers.convolution2d()
*       tf.contrib.layers.fully_connected()

#### Regularizers
*       tf.contrib.layers.l1_regularizer()
*       tf.contrib.layers.l2_regularizer()
*       tf.contrib.layers.sum_regularizer()

#### Initializers
*       tf.contrib.layers.xavier_initializer()
*       tf.contrib.layers.xavier_initializer_conv2d()

#### Summaries
*       tf.contrib.layers.summarize_activation()
*       tf.contrib.layers.summarize_tensor()
*       tf.contrib.layers.summarize_tensors()
*       tf.contrib.layers.summarize_collection()
*       tf.contrib.layers.summarize_activations()

#### Other Functions and Classes
*       tf.contrib.layers.apply_regularization()
*       tf.contrib.layers.legacy_convolution2d()
*       tf.contrib.layers.legacy_fully_connected()
*       tf.contrib.layers.make_all()
*       tf.contrib.layers.optimize_loss()
*       tf.contrib.layers.variance_scaling_initializer()


###  Utilities (contrib)

#### Miscellaneous Utility Functions
*       tf.contrib.util.constant_value()
*       tf.contrib.util.make_tensor_proto()
*       tf.contrib.util.make_ndarray()
*       tf.contrib.util.stripped_op_list_for_graph()

###  Wraps python functions

#### Script Language Operators
*       tf.py_func()
