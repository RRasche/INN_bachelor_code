import tensorflow as tf

def MMD_forward(y_true,y_pred):
    '''
    Input: 
        y_true <2D float array> contains data sampled from the prior distribution. Dimension: (batch_size,latent_dim + label_dim)
        y_pred <2D float array> contains data created by the network
    
    Output:
        <float> MMD^2(y_true,y_pred) Maximum Mean Discrepancy comparing the distributions of the data of both inputs
        
    Maximum Mean Discrepancy loss. The gradient is only calculated based on the first two dimensions of y_pred
    '''
    
    #splitting of the label and padding and stop the gradient in respect to it
    latent = y_pred[:,:2]
    label_and_padding = tf.stop_gradient(y_pred[:,2:])
    y_pred = tf.concat([latent,label_and_padding],1)
    
    #The 2D arrays can be unerstood as matrices. Then each column/row (depending on index choice) contains a full training dataset. 
    #Matrix multiplication is used to multiply and sum each possible combination of training datasets.
    #Code adapated from: https://github.com/VLL-HD/analyzing_inverse_problems/blob/master/toy_8-modes/toy_8-modes.ipynb function: MMD_multiscale
    xx, yy, xy = tf.matmul(y_pred,y_pred,transpose_b=True), tf.matmul(y_true,y_true,transpose_b=True), tf.matmul(y_pred,y_true,transpose_b=True)
    
    #tf.linalg.diag_part returns the diagonal as a vector
    #tf.broadcast_to fills a matrix of tf.shape(xx) with this vector.
    # [1, 2, 3] -> broadcast_to(2,3)
    #   [[1, 2, 3],
    #    [1, 2, 3]]
    rx = tf.broadcast_to(tf.linalg.diag_part(xx),tf.shape(xx))
    ry = tf.broadcast_to(tf.linalg.diag_part(yy),tf.shape(yy))
    
    #Calculating the L2 norm squared of every combination of training dataset
    #a^2 - 2*a*b + b^2,
    #where transpose(rx) contains a, 2.0*xx contains 2*a*b and rx contains b 
    dxx = tf.transpose(rx) + rx - 2.0*xx
    dyy = tf.transpose(ry) + ry - 2.*yy
    dxy = tf.transpose(rx) + ry - 2.*xy

    XX, YY, XY = tf.zeros_like(xx),tf.zeros_like(yy),tf.zeros_like(xy)

    #applaying the multi quadratic kernel. 
    #multiple values for a can be applied to get different gradients
    for a in [0.2, 0.05, 0.9]:
        XX += a**2 * (a**2 + dxx)**-1
        YY += a**2 * (a**2 + dyy)**-1
        XY += a**2 * (a**2 + dxy)**-1

    #the mean over everything is calculated to get MMD^2.
    return tf.reduce_mean(XX + YY - 2.*XY)
    
def MMD_backward(y_true,y_pred):
    '''
    Input: 
        y_true <2D float array> contains data sampled from the prior distribution. Dimension: (batch_size,latent_dim + label_dim)
        y_pred <2D float array> contains data created by the network
    
    Output:
        <float> MMD^2(y_true,y_pred) Maximum Mean Discrepancy comparing the distributions of the data of both inputs
        
    Maximum Mean Discrepancy loss.
    '''
    
    xx, yy, zz = tf.matmul(y_pred,y_pred,transpose_b=True), tf.matmul(y_true,y_true,transpose_b=True), tf.matmul(y_pred,y_true,transpose_b=True)
    rx = tf.broadcast_to(tf.linalg.diag_part(xx),tf.shape(xx))
    ry = tf.broadcast_to(tf.linalg.diag_part(yy),tf.shape(yy))

    dxx = tf.transpose(rx) + rx - 2.0*xx
    dyy = tf.transpose(ry) + ry - 2.*yy
    dxy = tf.transpose(rx) + ry - 2.*zz

    XX, YY, XY = tf.zeros_like(xx),tf.zeros_like(yy),tf.zeros_like(zz)

    for a in [0.2, 0.05, 0.9]:
        XX += a**2 * (a**2 + dxx)**-1
        YY += a**2 * (a**2 + dyy)**-1
        XY += a**2 * (a**2 + dxy)**-1

    return tf.reduce_mean(XX + YY - 2.*XY)
 