import 'package:flutter/material.dart';
import '../../WidgetsAndUtils/recipe_model.dart';
import '../admin_portal.dart';

class UpdateRecipe extends StatefulWidget {
  final String recipeID;
  final String recipeName;
  final String recipeCategory;
  final String recipeDiet;
  final String recipeCourse;
  final String recipeIngredients;
  final String recipeDescription;
  final String recipeTime;
  final int recipeRating;
  final String recipeURL;

  const UpdateRecipe({
    super.key,
    required this.recipeID,
    required this.recipeName,
    required this.recipeCategory,
    required this.recipeDiet,
    required this.recipeCourse,
    required this.recipeIngredients,
    required this.recipeDescription,
    required this.recipeTime,
    required this.recipeRating,
    required this.recipeURL,
  });

  @override
  State<UpdateRecipe> createState() => _UpdateRecipeState();
}

class _UpdateRecipeState extends State<UpdateRecipe> {
  int recipeID = 0;
  TextEditingController recipeNameTextController = TextEditingController();
  TextEditingController recipeCategoryTextController = TextEditingController();
  TextEditingController recipeDietTextController = TextEditingController();
  TextEditingController recipeCourseTextController = TextEditingController();
  TextEditingController recipeIngredientsTextController =
      TextEditingController();
  TextEditingController recipeDescriptionTextController =
      TextEditingController();
  TextEditingController recipeTimeTextController = TextEditingController();
  TextEditingController recipeURLTextController = TextEditingController();
  TextEditingController recipeRatingTextController = TextEditingController();

  @override
  void initState() {
    recipeNameTextController.text = widget.recipeName;
    recipeCategoryTextController.text = widget.recipeCategory;
    recipeDietTextController.text = widget.recipeDiet;
    recipeCourseTextController.text = widget.recipeCourse;
    recipeIngredientsTextController.text = widget.recipeIngredients;
    recipeDescriptionTextController.text = widget.recipeDescription;
    recipeTimeTextController.text = widget.recipeTime;
    recipeURLTextController.text = widget.recipeURL;
    recipeRatingTextController.text = widget.recipeRating.toString();
    super.initState();
  }

  Widget buildTextField(
      String labelText, TextEditingController textController) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 20.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            labelText,
            style: const TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.bold,
              color: Colors.black,
            ),
          ),
          TextField(
            maxLines: null,
            enabled: true,
            decoration: const InputDecoration(
              filled: false,
              floatingLabelBehavior: FloatingLabelBehavior.always,
              fillColor: Colors.black45,
            ),
            style: const TextStyle(
              color: Colors.black,
              height: 1.5,
            ),
            autofocus: false,
            controller: textController,
            keyboardType: TextInputType.multiline,
            textInputAction: TextInputAction.done,
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    Size size = MediaQuery.of(context).size;
    return Scaffold(
      appBar: AppBar(
        backgroundColor: const Color(0XFFFF4F5A),
        shape: const RoundedRectangleBorder(
          borderRadius: BorderRadius.only(
              bottomRight: Radius.circular(25),
              bottomLeft: Radius.circular(25)),
        ),
        centerTitle: true,
        title: const Text(
          'Update Recipes',
          textAlign: TextAlign.center,
        ),
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.only(left: 15.0, right: 15.0, top: 30.0),
          child:
              Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Row(
              children: [
                const Text(
                  'Recipe ID: ',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                Text(
                  widget.recipeID,
                  style: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.normal,
                  ),
                ),
              ],
            ),
            SizedBox(
              height: size.height * 0.04,
            ),
            SizedBox(
              height: size.height * 0.04,
            ),
            buildTextField('Recipe Name', recipeNameTextController),
            buildTextField('Recipe Ingredients(Comma Seprated)',
                recipeIngredientsTextController),
            buildTextField(
                'Recipe Instruction', recipeDescriptionTextController),
            buildTextField('Recipe Diet', recipeDietTextController),
            buildTextField('Recipe Course', recipeCourseTextController),
            buildTextField('Recipe Category', recipeCategoryTextController),
            buildTextField('Recipe Time', recipeTimeTextController),
            buildTextField('Recipe Rating', recipeRatingTextController),
            buildTextField('Recipe Image URL', recipeURLTextController),
            Container(
              width: MediaQuery.of(context).size.width,
              height: 40,
              margin: const EdgeInsets.fromLTRB(0, 10, 0, 10),
              decoration:
                  BoxDecoration(borderRadius: BorderRadius.circular(90)),
              child: ElevatedButton(
                onPressed: () {
                  RecipeModel().UpdateRecipe(
                    recipeID: int.parse(widget.recipeID),
                    recipeCategory: recipeCategoryTextController.text,
                    recipeCourse: recipeCourseTextController.text,
                    recipeDiet: recipeDietTextController.text,
                    recipeRating: recipeRatingTextController.text,
                    recipeName: recipeNameTextController.text,
                    recipeDescription: recipeDescriptionTextController.text,
                    recipeIngredients: recipeIngredientsTextController.text,
                    recipeImageURL: recipeURLTextController.text,
                    recipeTime: recipeTimeTextController.text,
                  );
                  Navigator.pushAndRemoveUntil(
                    context,
                    PageRouteBuilder(
                      transitionDuration: const Duration(seconds: 1),
                      transitionsBuilder:
                          (context, animation, animationTime, child) {
                        animation = CurvedAnimation(
                            parent: animation,
                            curve: Curves.fastLinearToSlowEaseIn);
                        return ScaleTransition(
                          scale: animation,
                          alignment: Alignment.center,
                          child: child,
                        );
                      },
                      pageBuilder: (context, animation, animationTime) {
                        return AdminPortal();
                      },
                    ),
                    (route) => false,
                  );
                },
                style: ButtonStyle(
                    backgroundColor:
                        MaterialStateProperty.resolveWith((states) {
                      if (states.contains(MaterialState.pressed)) {
                        return Colors.blueGrey;
                      }
                      return const Color(0XFFFF4F5A);
                    }),
                    shape: MaterialStateProperty.all(RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(30)))),
                child: const Text(
                  'Update',
                  style: TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.bold,
                      fontSize: 16),
                ),
              ),
            ),
          ]),
        ),
      ),
    );
  }
}
