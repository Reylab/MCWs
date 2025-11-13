-----------------------
 MCWs pipeline runs from 
"processing_steps_MCW_reduced"
-----------------------

this function contains all relevant subfunctions
to use together or independently

- As is this is set up to run in parallel
	however that is a variable you can set to false

- Saving or creating plots as pngs/matlab figs can be turned off in
	parameter inputs
 make sure matlab directory has code added to path list
	 
----------------
if starting with NSX file

- recommended to run all the way through
- only need to add your personal computers path
- make sure matlab directory has code added to path
	- directory is set to be where your NSX file is

Functions:
- NSX parsing
- NCX channel reading 
- spike detection
- collison detection: collision in this instance is when an
	event is seen across multiple channels - likely an 
	artifact interfering with the channels
- quarantine spikes: this is also called artifact removal
	- this makes a mask for spikes that do not have 
	shapes deemed good by criteria defined from the peaks
	- these spikes are then reviewed later once the better
	spikes are clustered
- clustering:
	- clusters using a GMM feature extraction
		NOTE: the gmm requires distribution fitting
		this can take some time to run on personal 
		laptops 
		NOTE: this only needs to be done once then the 
		fit distributions are saved 
- metric print out from clustering and spike shapes
- rescue mask takes the quarantine spikes and sees if they fall into any clusters
- metrics for new spikes printed in same format

-------------------
For use with NCX or single channel
- can start at line 176
- rest of functions remain the same
