from django import forms

class ApprovalForm(forms.Form):
	GENDER_CHOICES = [
		('Male', 'Male'),
		('Female', 'Female')
	]
	YES_NO_CHOICES = [
		('Yes', 'Yes'),
		('No', 'No')
	]
	REGION = [
		('southwest', 'southwest'),
		('northwest', 'northwest'),
		('northeast', 'northeast'),
		('southeast','southeast')
	]
	firstname=forms.CharField(max_length=15)
	lastname=forms.CharField(max_length=15)
	age=forms.IntegerField()
	gender=forms.ChoiceField(choices=GENDER_CHOICES)
	bmi=forms.FloatField()
	children=forms.IntegerField()
	region=forms.ChoiceField( choices=REGION)
	smoker=forms.ChoiceField(choices=YES_NO_CHOICES)
	alcoholConsumer=forms.ChoiceField(choices=YES_NO_CHOICES)
	diphtheria=forms.ChoiceField(choices=YES_NO_CHOICES)
	polio=forms.ChoiceField(choices=YES_NO_CHOICES)
	measles=forms.ChoiceField(choices=YES_NO_CHOICES)
	hepatitis=forms.ChoiceField(choices=YES_NO_CHOICES)
	hiv_aids=forms.ChoiceField(choices=YES_NO_CHOICES)
	thinness=forms.ChoiceField(choices=YES_NO_CHOICES)
	
