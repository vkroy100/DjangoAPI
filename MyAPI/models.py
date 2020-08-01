from django.db import models

class approvals(models.Model):
	GENDER_CHOICES = (
		('Male', 'Male'),
		('Female', 'Female')
	)
	YES_NO_CHOICES = (
		('Yes', 'Yes'),
		('No', 'No')
	)
	REGION = (
		('southwest', 'southwest'),
		('northwest', 'northwest'),
		('northeast', 'northeast'),
		('southeast', 'southeast')
	)
	firstname=models.CharField(max_length=15)
	lastname=models.CharField(max_length=15)
	age=models.IntegerField(default=0)
	gender=models.CharField(max_length=15, choices=GENDER_CHOICES)
	bmi=models.FloatField(default=0)
	children=models.IntegerField(default=0)
	region=models.CharField(max_length=15, choices=REGION)
	smoker=models.CharField(max_length=15,choices=YES_NO_CHOICES)
	alcoholConsumer=models.CharField(max_length=15,choices=YES_NO_CHOICES)
	diphtheria=models.CharField(max_length=15,choices=YES_NO_CHOICES)
	polio=models.CharField(max_length=15,choices=YES_NO_CHOICES)
	measles=models.CharField(max_length=15,choices=YES_NO_CHOICES)
	hepatitis=models.CharField(max_length=15,choices=YES_NO_CHOICES)
	hiv_aids=models.CharField(max_length=15,choices=YES_NO_CHOICES)
	thinness=models.CharField(max_length=15,choices=YES_NO_CHOICES)
	def __str__(self):
		return '{}, {}'.format(self.lastname, self.firstname)
