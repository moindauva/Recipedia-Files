class Recipes {
  String recipeID;
  String recipeName;
  String recipeDescription;
  String recipeURL;
  String recipeRating;
  String recipeCategory;
  String recipeTime;
  List<String> recipeIngredients = [];

  Recipes({
    required this.recipeID,
    required this.recipeName,
    required this.recipeDescription,
    required this.recipeURL,
    required this.recipeRating,
    required this.recipeCategory,
    required this.recipeTime,
    required this.recipeIngredients,
  });
}
