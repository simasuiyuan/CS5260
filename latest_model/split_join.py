#%%
import pickle
# %%
def split(source, dest_folder, write_size):
    # Make a destination folder if it doesn't exist yet
    if not os.path.exists(dest_folder):
        os.mkdir(dest_folder)
    else:
        # Otherwise clean out all files in the destination folder
        for file in os.listdir(dest_folder):
            os.remove(os.path.join(dest_folder, file))
    partnum = 0
    
    # Open the source file in binary mode
    input_file = open(source, 'rb')
    while True:
        # Read a portion of the input file
        chunk = input_file.read(write_size)
        
        # End the loop if we have hit EOF
        if not chunk:
            break
        
        # Increment partnum
        partnum += 1
        
        # Create a new file name
        filename = f'{dest_folder}/final_model' + str(partnum)
        
        # Create a destination file
        dest_file = open(filename, 'wb')
        
        # Write to this portion of the destination file
        dest_file.write(chunk)
        # Explicitly close 
        dest_file.close()
    # Explicitly close
    input_file.close()
    # Return the number of files created by the split
    return partnum
# %%
split(source='decoder-57.pkl', write_size=20000000, dest_folder='./decoder-57')
split(source='encoder-57.pkl', write_size=20000000, dest_folder='./encoder-57')
# %%
def join(source_dir, dest_file, read_size):
    # Create a new destination file
    output_file = open(dest_file, 'wb')
     
    # Get a list of the file parts
    parts = ['final_model1','final_model2','final_model3']
 
    # Go through each portion one by one
    for file in parts:
         
        # Assemble the full path to the file
        path = file
         
        # Open the part
        input_file = open(path, 'rb')
         
        while True:
            # Read all bytes of the part
            bytes = input_file.read(read_size)
             
            # Break out of loop if we are at end of file
            if not bytes:
                break
                 
            # Write the bytes to the output file
            output_file.write(bytes)
             
        # Close the input file
        input_file.close()
         
    # Close the output file
    output_file.close()
join(source_dir='', dest_file="Combined_Model.p", read_size = 50000000)