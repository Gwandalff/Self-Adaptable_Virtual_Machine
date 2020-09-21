package visitor.operation.miniJava.impl;

import java.lang.Boolean;
import java.lang.Integer;
import miniJava.visitor.miniJava.Clazz;
import miniJava.visitor.miniJava.Member;
import miniJava.visitor.miniJava.Method;
import miniJava.visitor.miniJava.Parameter;
import miniJava.visitor.miniJava.State;
import miniJava.visitor.miniJava.TypeRef;
import minijava.MapService;
import org.eclipse.emf.common.util.EMap;
import org.eclipse.emf.ecoretools.ale.compiler.lib.CollectionService;
import org.eclipse.emf.ecoretools.ale.compiler.lib.EqualService;
import visitor.VisitorInterface;
import visitor.operation.miniJava.BlockOperation;
import visitor.operation.miniJava.MethodOperation;
import visitor.operation.miniJava.ParameterOperation;
import visitor.operation.miniJava.TypeRefOperation;

public class MethodOperationImpl extends MemberOperationImpl implements MethodOperation {
	private final Method it;

	private final VisitorInterface vis;

	public MethodOperationImpl(Method it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}

	public void evaluateStatement(State state) {
		((BlockOperation)this.it.getBody().accept(vis)).evaluateStatement((State) (state));
	}

	public Method findOverride(Clazz c) {
		Method result;
		if(!(MapService.containsKey((EMap) (this.it.getCache()), (Clazz) (c)))) {
			Method that = ((Method) (this.it));
			if(CollectionService.exists(c.getMembers(), (x) -> EqualService.equals((x), (that)))) {
				result = ((Method) (this.it));
			}
			else {
				int i = ((int) (0));
				Method found = ((Method) (null));
				while ((((i) < (CollectionService.size(c.getMembers()))) && (EqualService.equals((found), (null))))) {
					Member tmpm = ((Member) (CollectionService.get(c.getMembers(), i)));
					if(tmpm instanceof Method) {
						Method m = ((Method) (tmpm));
						if(((EqualService.equals((m.getName()), (this.it.getName()))) && (EqualService.equals((CollectionService.size(m.getParams())), (CollectionService.size(this.it.getParams())))))) {
							boolean compared = ((boolean) (((TypeRefOperation)m.getTypeRef().accept(vis)).compare((TypeRef) (this.it.getTypeRef()))));
							int j = ((int) (0));
							int paramlgt = ((int) (CollectionService.size(m.getParams())));
							boolean alltrue = ((boolean) (compared));
							while ((((j) < (paramlgt)) && (alltrue))) {
								Parameter p1 = ((Parameter) (CollectionService.get(m.getParams(), j)));
								Parameter tmpp = ((Parameter) (CollectionService.head(CollectionService.select(this.it.getParams(), (p2) -> ((ParameterOperation)p1.accept(vis)).compare((Parameter) (p2))))));
								alltrue = ((Boolean) (!EqualService.equals((tmpp), (null))));
								j = ((Integer) ((j) + (1)));
							}
							if(alltrue) {
								found = ((Method) (m));
							}
						}
					}
					i = ((Integer) ((i) + (1)));
				}
				if(!EqualService.equals((found), (null))) {
					result = ((Method) (found));
				}
				else {
					if(!EqualService.equals((c.getSuperClass()), (null))) {
						result = ((Method) (((MethodOperation)this.it.accept(vis)).findOverride((Clazz) (c.getSuperClass()))));
					}
					else {
						result = ((Method) (null));
					}
				}
			}
			MapService.put((EMap) (this.it.getCache()), (Clazz) (c), (Method) (result));
		}
		else {
			result = ((Method) (MapService.getFromMap((EMap) (this.it.getCache()), (Clazz) (c))));
		}
		return result;
	}

	public void call(State state) {
		((BlockOperation)this.it.getBody().accept(vis)).evaluateStatement((State) (state));
	}
}
