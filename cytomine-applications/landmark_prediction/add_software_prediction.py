# -*- coding: utf-8 -*-

#
# * Copyright (c) 2009-2015. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# */


__author__          = "Vandaele Rémy <remy.vandaele@ulg.ac.be>"
__contributors__    = ["Marée Raphaël <raphael.maree@ulg.ac.be>"]
__copyright__       = "Copyright 2010-2015 University of Liège, Belgium, http://www.cytomine.be/"

import cytomine

#connect to cytomine : parameters to set
cytomine_host="" #eg: "demo.cytomine.be"
cytomine_public_key=""
cytomine_private_key=""
id_project=0 #Cytomine project id

#Connection to Cytomine Core
conn = cytomine.Cytomine(cytomine_host, cytomine_public_key, cytomine_private_key, base_path = '/api/', working_path = '/tmp/', verbose= True)


#Generic
software = conn.add_software("Landmark_Generic_Prediction", "pyxitSuggestedTermJobService","ValidateAnnotation")
conn.add_software_parameter("cytomine_models",             software.id, "String", None, True, 1 , False)
conn.add_software_parameter("cytomine_predict_images",     software.id, "String", None, True, 2 , False)
conn.add_software_parameter("prediction_error",            software.id, "String", None, True, 3 , False)
conn.add_software_project(id_project,software.id)
print "Generic Prediction Software id is %d"%software.id

#DMBL
software = conn.add_software("Landmark_DMBL_Prediction", "pyxitSuggestedTermJobService","ValidateAnnotation")
conn.add_software_parameter("cytomine_models",             software.id, "String", None, True, 1 , False)
conn.add_software_parameter("cytomine_predict_images",     software.id, "String", None, True, 2 , False)
conn.add_software_parameter("prediction_error",            software.id, "String", None, True, 3 , False)
conn.add_software_project(id_project,software.id)
print "DMBL Prediction Software id is %d"%software.id

#LC
software = conn.add_software("Landmark_LC_Prediction", "pyxitSuggestedTermJobService","ValidateAnnotation")
conn.add_software_parameter("cytomine_models",             software.id, "String", None, True, 1 , False)
conn.add_software_parameter("cytomine_predict_images",     software.id, "String", None, True, 2 , False)
conn.add_software_parameter("prediction_error",            software.id, "String", None, True, 3 , False)
conn.add_software_project(id_project,software.id)
print "LC Prediction Software id is %d"%software.id
