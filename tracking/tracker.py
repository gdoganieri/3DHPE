import numpy as np
from tracking.kalmanFilter import KalmanFilter
from scipy.optimize import linear_sum_assignment
from collections import deque
import random

class Tracks(object):
	"""docstring for Tracks"""
	def __init__(self, detection, trackId):
		super(Tracks, self).__init__()
		self.KF = KalmanFilter()
		self.KF.predict()
		self.KF.correct(np.matrix(detection).reshape(3,1))
		self.trace = deque(maxlen=20)
		self.prediction = detection.reshape(1,3)
		self.trackId = trackId
		self.skipped_frames = 0
		self.track_color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
		self.track_bbox = []

	def predict(self,detection):
		self.prediction = np.array(self.KF.predict()).reshape(1,3)
		self.KF.correct(np.matrix(detection).reshape(3,1))


class Tracker(object):
	"""docstring for Tracker"""
	def __init__(self, dist_threshold, max_frame_skipped, max_trace_length):
		super(Tracker, self).__init__()
		self.dist_threshold = dist_threshold
		self.max_frame_skipped = max_frame_skipped
		self.max_trace_length = max_trace_length
		self.trackId = 0
		self.tracks = []
		self.del_tracks = []
		self.un_assigned_tracks = []


	def update(self, detections):
		if len(self.tracks) == 0:
			for i in range(detections.shape[0]):
				track = Tracks(detections[i], self.trackId)
				self.trackId +=1
				self.tracks.append(track)

		N = len(self.tracks)
		M = len(detections)
		cost = []
		for i in range(N):
			diff = np.linalg.norm(self.tracks[i].prediction - detections.reshape(-1,3), axis=1)
			cost.append(diff)

		cost = np.array(cost)*0.1
		row, col = linear_sum_assignment(cost)
		assignment = [-1]*N
		for i in range(len(row)):
			assignment[row[i]] = col[i]



		for i in range(len(assignment)):
			if assignment[i] != -1:
				if cost[i][assignment[i]] > self.dist_threshold:
					assignment[i] = -1
					self.un_assigned_tracks.append(i)
				else:
					self.tracks[i].skipped_frames += 1


		for i in range(len(self.tracks)):
			if self.tracks[i].skipped_frames > self.max_frame_skipped :
				self.del_tracks.append(i)

		if len(self.del_tracks) > 0:
			for i in range(len(self.del_tracks)):
				del self.tracks[i]
				del assignment[i]
				self.del_tracks = []

		for i in range(len(detections)):
			if i not in assignment:
				track = Tracks(detections[i], self.trackId)
				self.trackId +=1
				self.tracks.append(track)


		for i in range(len(assignment)):
			if(assignment[i] != -1):
				self.tracks[i].skipped_frames = 0
				self.tracks[i].predict(detections[assignment[i]])
			self.tracks[i].trace.append(self.tracks[i].prediction)