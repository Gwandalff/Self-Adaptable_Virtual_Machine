package miniJava.interpreter.dynamicModules;

import miniJava.interpreter.IDynamicModule;
import miniJava.interpreter.IDynamicSubject;
import miniJava.interpreter.miniJava.Expression;
import miniJava.interpreter.miniJava.MiniJavaPackage;
import miniJava.interpreter.miniJava.PrintStatement;
import miniJava.interpreter.miniJava.StringConstant;
import miniJava.interpreter.miniJava.Value;

public class DynamicSemanticInjectionModule implements IDynamicModule {

	private long timeMillis = 0;
	
	@Override
	public boolean updateBefore(IDynamicSubject self, Object[] args) {
		try {
			Expression exp = ((PrintStatement) self).getExpression();
			String print = ((StringConstant) exp).getValue();
			if(print.startsWith("-> ")) {
				print = print.substring(3);
				switch (print) {
				case "tic":
					timeMillis = System.currentTimeMillis();
					break;
				case "toc":
					timeMillis = System.currentTimeMillis() - timeMillis;
					break;
				case "millis":
					System.out.println(timeMillis + "ms");
					break;
				case "seconds":
					System.out.println((timeMillis/1000) + "s");
					break;
				}
				return false;
			} else {
				return true;
			}
		} catch (Exception e) {
			return true;
		}
	}

	@Override
	public boolean updateAfter(IDynamicSubject self, Object[] args, Value returned) {
		return false;
	}

	@Override
	public Value byPassResult() {
		return null;
	}

	@Override
	public int[] targetedNodes() {
		int[] out = {MiniJavaPackage.PRINT_STATEMENT};
		return out;
	}

	@Override
	public Strategy callStrategy() {
		return Strategy.AROUND;
	}

}
