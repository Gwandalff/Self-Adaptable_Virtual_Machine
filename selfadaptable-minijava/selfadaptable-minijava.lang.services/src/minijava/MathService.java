package minijava;

public class MathService {
	public static Integer mod(Object self, Integer lhs, Integer rhs) {
		return lhs % rhs;
	}
	
	public static Integer sqrt(Object self, Integer expression) {
		return (int) Math.sqrt(expression);
	}
}
